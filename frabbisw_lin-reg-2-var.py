import torch
import torch.nn as nn
import numpy as np
def sample(n):
    #x=np.array([[5.,3.1],[2.,1.2],[3.,4.3],[4.,2.4],[1.,5.5],],dtype='float32')
    #np.random.seed(420)
    x=np.random.uniform(0,5,2*n).reshape(n,2).astype('float32')
    y=3.1*x[:,0]-2.5*x[:,1]+.4
    y=y.reshape((n,1))
    
    x=torch.from_numpy(x).cuda()
    y=torch.from_numpy(y).cuda()
    
    return x,y
#y
x,y=sample(10)
model=nn.Linear(2,1).cuda()
optim=torch.optim.Adam(model.parameters(),.01)
loss_fn = torch.nn.MSELoss()
for i in range(1000):
    x,y=sample(1000)
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if i % 100 == 0:
        print(loss)

#print(y)
#print(y_pred)
