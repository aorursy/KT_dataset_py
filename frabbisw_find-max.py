import numpy as np # linear algebra
import torch
import torch.nn as nn
def sample(n):
    x=np.random.uniform(0,1,n*10).reshape(n,10).astype('float32')
    x=np.sort(x)
    y=np.max(x,axis=1).reshape(n,1)
    return torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
class Model(nn.Module):
    def __init__(self,din,dout):
        super(Model,self).__init__()
        self.linear1=nn.Linear(din,15)
        self.linear2=nn.Linear(15,20)
        self.linear3=nn.Linear(20,10)
        self.linear4=nn.Linear(10,5)
        self.linear5=nn.Linear(5,dout)
    def forward(self,x):
        y=torch.tanh(self.linear1(x))
        y=torch.tanh(self.linear2(y))
        y=torch.tanh(self.linear3(y))
        y=torch.tanh(self.linear4(y))
        y=self.linear5(y)
        return y
model=Model(10,1).cuda()
optim=torch.optim.SGD(model.parameters(),lr=.001)
criterion=nn.MSELoss()
for i in range(1000000):
    x,y=sample(10000)
    z=model(x)
    loss=criterion(z,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 1000 == 0:
        print(loss)
x,y=sample(1)
z=model(x)
print("real:\t\t",y.cpu().detach().numpy()[0,0],"\npredicted:\t",z.cpu().detach().numpy()[0,0])
torch.save(model.state_dict(), "max.pt")
'''
model=Model(10,1).cuda()
model.load_state_dict(torch.load("max.pt"))
model.eval()
'''
