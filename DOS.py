import numpy as np
import torch
import torch.nn as nn
np.random.seed(234198)

import scipy.stats

class stock:
    def __init__(self, T, K, sigma, delta, So, r, N, M, d):
        self.T = T
        self.K=K
        self.sigma=sigma *np.ones(d)
        self.delta=delta
        self.So=So*np.ones(d)
        self.r=r
        self.N=N
        self.M=M
        self.d=d
    
    def GBM(self):
        
        dt=self.T/self.N
        So_vec=self.So*np.ones((1,S.M, S.d))
        
        Z=np.random.standard_normal((self.N,self.M, self.d))
        s=self.So*np.exp(np.cumsum((self.r-self.delta-0.5*self.sigma**2)*dt+self.sigma*np.sqrt(dt)*Z, axis=0))
        
        s=np.append(So_vec, s, axis=0)
        return s
    
    
    def g(self,n,m,X):
        max1=torch.max(X[int(n),m,:].float()-self.K)
        
        return np.exp(-self.r*(self.T/self.N)*n)*torch.max(max1,torch.tensor([0.0]))
       

#%%
class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2):
        super(NeuralNet, self).__init__()
        self.a1 = nn.Linear(d, q1) 
        self.relu = nn.ReLU()
        self.a2 = nn.Linear(q1, q2)
        self.a3 = nn.Linear(q2, 1)  
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        out = self.a1(x)
        out = self.relu(out)
        out = self.a2(out)
        out = self.relu(out)
        out = self.a3(out)
        out = self.sigmoid(out)
        
        return out
    
def loss(y_pred,s, x, n, tau):
    r_n=torch.zeros((s.M))
    for m in range(0,s.M):
        
        r_n[m]=-s.g(n,m,x)*y_pred[m] - s.g(tau[m],m,x)*(1-y_pred[m])
    
    return(r_n.mean())
    
#%%

S=stock(3,100,0.2,0.1,90,0.05,9,5000,10)   

X=torch.from_numpy(S.GBM()).float()
#%%

def NN(n,x,s, tau_n_plus_1):
    epochs=50
    model=NeuralNet(s.d,s.d+40,s.d+40)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    for epoch in range(epochs):
        F = model.forward(X[n])
        optimizer.zero_grad()
        criterion = loss(F,S,X,n,tau_n_plus_1)
        criterion.backward()
        optimizer.step()
    
    return F,model

mods=[None]*S.N
tau_mat=np.zeros((S.N+1,S.M))
tau_mat[S.N,:]=S.N

f_mat=np.zeros((S.N+1,S.M))
f_mat[S.N,:]=1

#%%
for n in range(S.N-1,-1,-1):
    probs, mod_temp=NN(n, X, S,torch.from_numpy(tau_mat[n+1]).float())
    mods[n]=mod_temp
    np_probs=probs.detach().numpy().reshape(S.M)
    print(n, ":", np.min(np_probs)," , ", np.max(np_probs))

    f_mat[n,:]=(np_probs > 0.5)*1.0

    tau_mat[n,:]=np.argmax(f_mat, axis=0)

#%% 
Y=torch.from_numpy(S.GBM()).float()

tau_mat_test=np.zeros((S.N+1,S.M))
tau_mat_test[S.N,:]=S.N

f_mat_test=np.zeros((S.N+1,S.M))
f_mat_test[S.N,:]=1

V_mat_test=np.zeros((S.N+1,S.M))
V_est_test=np.zeros(S.N+1)

for m in range(0,S.M):
    V_mat_test[S.N,m]=S.g(S.N,m,Y)
    
V_est_test[S.N]=np.mean(V_mat_test[S.N,:])



for n in range(S.N-1,-1,-1):
    mod_curr=mods[n]
    probs=mod_curr(Y[n])
    np_probs=probs.detach().numpy().reshape(S.M)

    f_mat_test[n,:]=(np_probs > 0.5)*1.0

    tau_mat_test[n,:]=np.argmax(f_mat_test, axis=0)
    
    
    for m in range(0,S.M):
        V_mat_test[n,m]=np.exp((n-tau_mat_test[n,m])*(-S.r*S.T/S.N))*S.g(tau_mat_test[n,m],m,X) 
        

#%%
V_est_test=np.mean(V_mat_test, axis=1)
V_std_test=np.std(V_mat_test, axis=1)
V_se_test=V_std_test/(np.sqrt(S.M))

z=scipy.stats.norm.ppf(0.975)
lower=V_est_test[0] - z*V_se_test[0]
upper=V_est_test[0] + z*V_se_test[0]

print(V_est_test[0])
print(V_se_test[0])
print(lower)
print(upper)