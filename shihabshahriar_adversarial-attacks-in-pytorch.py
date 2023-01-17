import numpy as np # linear algebra

import pandas as pd

from sklearn.metrics import *

from PIL import Image

from io import BytesIO

import requests



import torch

import torch.nn as nn 

import torch.nn.functional as F

import torch.optim as optim 

from torchvision.transforms import ToTensor

from torchvision.utils import make_grid

import matplotlib.pyplot as plt
data = pd.read_csv("../input/mnist_train.csv")

X,Y = data.values[:,1:]/255,data.values[:,0]

X = torch.from_numpy(X).cuda().view(-1,1,28,28)

Y = torch.from_numpy(Y).cuda()
data = pd.read_csv("../input/mnist_test.csv")

Xtest,Ytest = data.values[:,1:]/255,data.values[:,0]

Xtest = torch.from_numpy(Xtest).cuda().view(-1,1,28,28)

Ytest = torch.from_numpy(Ytest).cuda()
class LeNet(nn.Module):

    def __init__(self):

        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, 1)

        self.conv2 = nn.Conv2d(16, 32, 5, 1)

        self.fc1 = nn.Linear(4*4*32, 256)

        self.fc2 = nn.Linear(256, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*32)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

model = LeNet().cuda().double()

opt = optim.Adam(model.parameters(),lr=.001)

crit = nn.CrossEntropyLoss()

BATCH_SIZE = 100
%%time

avg = []

XX = X.view(-1,1,28,28)

for i in range(6000):

    inp_idx = torch.randperm(len(X))[:BATCH_SIZE]

    XB = XX[inp_idx]

    YB = Y[inp_idx]

    

    opt.zero_grad()

    yp = model(XB)

    loss = crit(yp,YB)

    loss.backward()

    opt.step()

    

    avg.append(loss.item())

    if i%600==0:

        #lr.step()

        print(i//600,sum(avg)/len(avg))

        avg = []
start = 0

outs = []

while start<len(Xtest):

    XB = Xtest[start:start+200]

    with torch.no_grad():

        outs.append(model(XB))

    start += 200

YP = torch.cat(outs,dim=0)

yp = torch.argmax(YP,dim=1)

'Test Accuarcy:',(Ytest==yp).sum().item()/len(Ytest)*100
idx = 999

plt.imshow(Xtest[idx].cpu().numpy().transpose(1,2,0).reshape(28,28));Ytest[idx].item()
y = model(Xtest[idx].unsqueeze(0)).detach()

y,torch.argmax(y)
x = Xtest[idx].clone()

x.requires_grad_(True)

with torch.no_grad():

    logits = model(x.unsqueeze(0)).squeeze()

    IMX = torch.argmax(logits)

print("Init idx:",IMX.item())

    

lr=.01

while True:

    logits = model(x.unsqueeze(0)).squeeze()

    

    imx = torch.argmax(logits)

    if imx!=IMX: 

        print("Job done, breaking")

        break

    y = logits.clone()

    y[imx] = -99

    loss = logits.max() - y.max() 

    loss.backward()

    

    

    x.data.sub_(lr*x.grad.data)

    print(loss.item(),logits.max().item(), y.max().item() )

    print()

    x.grad.data.zero_()

    x.data.sub_(x.data.min())

    x.data.mul_(1/x.data.max())

    

with torch.no_grad():

    print(model(x.unsqueeze(0)).squeeze())
plt.imshow(x.detach().cpu().numpy().transpose(1,2,0).reshape(28,28));torch.argmax(model(x.unsqueeze(0))).item()

x = Xtest[idx].clone()

x.requires_grad_(True)

with torch.no_grad():

    logits = model(x.unsqueeze(0)).squeeze()

    IMX = torch.argmax(logits)

print("Init idx:",IMX.item())





logits = model(x.unsqueeze(0))

loss = F.cross_entropy(logits,Ytest[idx].unsqueeze(0))

loss.backward()

x = x + .06 * torch.sign(x.grad.data)

with torch.no_grad():

    logits = model(x.unsqueeze(0)).squeeze()

    print(logits,torch.argmax(logits))
plt.imshow(x.detach().cpu().numpy().transpose(1,2,0).reshape(28,28));torch.argmax(model(x.unsqueeze(0))).item()
































