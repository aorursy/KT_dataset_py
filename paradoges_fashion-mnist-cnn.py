print(1)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

path="../input/"
train=pd.read_csv(path+'fashion-mnist_train.csv')
test=pd.read_csv(path+'fashion-mnist_test.csv')

bs=100
epoch=100
lr=0.01

train_y=torch.from_numpy(np.array(train['label']).reshape(-1,bs))
train_x=torch.from_numpy(np.array(train.iloc[:,1:]).reshape(-1,bs,784))

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.maxpool=2
        self.ks_conv=5
        self.padding=2
        self.conv_step=1
        self.inchn_conv1=1
        self.inchn_conv2=16
        self.outchn_conv2=32
        self.conv1=nn.Sequential(nn.Conv2d(self.inchn_conv1,self.inchn_conv2,self.ks_conv,stride=self.conv_step,padding=self.padding),
                            nn.BatchNorm2d(self.inchn_conv2),
                            nn.ReLU(),
                            nn.MaxPool2d(self.maxpool))
        self.conv2=nn.Sequential(nn.Conv2d(self.inchn_conv2,self.outchn_conv2,self.ks_conv,stride=self.conv_step,padding=self.padding),
                            nn.BatchNorm2d(self.outchn_conv2),
                            nn.ReLU(),
                            nn.MaxPool2d(self.maxpool))
        self.out=nn.Linear(32*7*7,10)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output

cnn=CNN()
optimizer = torch.optim.Adam(cnn.parameters(),lr=lr)
loss_func = nn.CrossEntropyLoss()

for e in range(0,epoch):
    for step in range(0,60000//bs):
        x=Variable(train_x[step].view(bs,1,28,28).float())
        y=Variable(train_y[step].long())
        output = cnn(x) 
        loss = loss_func(output,y)
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        print('Epoch: ', e, 'Batch: ', step, '| train loss: %.4f' % loss.data.item())
test_x=Variable(torch.Tensor(np.array(test.iloc[:,1:])).view(-1,1,28,28))
output=cnn(test_x)
y = torch.max(output,1)[1].data.numpy().squeeze()
y.shape
test_y=np.array(test['label'])
total=0
pt=0
for i in range(0,10000):
    total+=1
    if test_y[i]==y[i]:
        pt+=1
accuracy=pt/total
print(accuracy)