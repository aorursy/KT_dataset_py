import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.nn.modules import Linear
from torch.nn.modules import Sequential
from random import uniform as rf
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def get_sample(n):
    x=np.random.uniform(-1,1,n*2).reshape(n,2)
    y=np.where(np.sqrt(np.sum(np.square(x),axis=1))<0.75,1,0).reshape(n,1)
    #y=.5*np.sum(x,axis=1)
    return torch.from_numpy(x.astype('float32')).cuda(),torch.from_numpy(y.astype('float32')).cuda()
class Model(nn.Module):
    def __init__(self, din, dout):
        super(Model, self).__init__()
        self.linear1=Linear(din,5)
        self.linear2=Linear(5,3)
        self.linear3=Linear(3,1)
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.LogSoftmax(dim=1)
    def forward(self, x):
        y=self.tanh(self.linear1(x))
        y=self.tanh(self.linear2(y))
        y=self.sigmoid(self.linear3(y))
        return y
model=Model(2,1).cuda()
optim=Adam(model.parameters(),.0001)
def train(n):
    x,y=get_sample(n)
    y_pred=model(x)
    loss=F.mse_loss(y_pred,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    return loss,y,y_pred
for i in range(1000000):
    loss,y_real,y_pred=train(10000)
    if i % 100000 == 0:
        print(loss)
x,y=get_sample(1000)
y_pred=model(x)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x.cpu().numpy()[:,0:1],x.cpu().numpy()[:,1:2],y.cpu().detach().numpy())
plt.show()

fig2 = plt.figure()
ax = Axes3D(fig2)

ax.scatter(x.cpu().numpy()[:,0:1],x.cpu().numpy()[:,1:2],y_pred.cpu().detach().numpy())
plt.show()
