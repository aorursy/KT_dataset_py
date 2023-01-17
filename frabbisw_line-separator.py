import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
def get_sample(n):
    x=np.random.uniform(-5,5,n).reshape(n,1)
    y=np.where(x<0,0,1)
    return torch.from_numpy(x.astype('float32')).cuda(),torch.from_numpy(y.astype('float32')).cuda()
class Model(nn.Module):
    def __init__(self,din,dout):
        super(Model, self).__init__()
        self.linear1=nn.Linear(din,dout)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        y=self.sigmoid(self.linear1(x))
        return y
model=Model(1,1).cuda()
optim=torch.optim.SGD(model.parameters(),.03)
def train(n):
    x,y=get_sample(n)
    y_pred=model(x)
    loss=F.mse_loss(y_pred,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    return loss,y,y_pred
def test(n):
    x,y=get_sample(n)
    y_pred=model(x)
    plt.scatter(x.cpu().numpy(),y_pred.cpu().detach().numpy())
    plt.show()
test(1000)
for i in range(10000):
    loss,y,y_pred=train(10000)
    if i % 500 == 0:
        print(loss)
    if i % 1000 == 0:
        test(1000)

