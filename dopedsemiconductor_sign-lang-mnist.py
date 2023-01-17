# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
traindata = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')
testdata = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
print(traindata.shape,testdata.shape)
trainlab = traindata['label'].values
testlab = testdata['label'].values
traindata = (traindata.iloc[:,1:].values).astype('float32')
testdata = (testdata.iloc[:,1:].values).astype('float32')
traindata = traindata.reshape(traindata.shape[0],1,28,28)
testdata = testdata.reshape(testdata.shape[0],1,28,28)
print(traindata.shape,testdata.shape,trainlab.shape,testlab.shape)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
trainx = torch.tensor(traindata)/255.0
trainy = torch.tensor(trainlab)
testx = torch.tensor(testdata)/255.0
testy = torch.tensor(testlab)
train = TensorDataset(trainx, trainy)
test = TensorDataset(testx, testy)
train_loader = DataLoader(train, batch_size=16, num_workers=2, shuffle=True)
test_loader = DataLoader(test, batch_size=16, num_workers=2, shuffle=False)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.s = 1
        self.c1 = nn.Conv2d(1,32,3)
        self.c2 = nn.Conv2d(32,64,3)
        self.c3 = nn.Conv2d(64,128,3)
        
        self.f2 = nn.Linear(512,26)
    def forward(self, x):
        x = self.c1(x)
        x = F.max_pool2d(F.relu(x),(2,2))
        x = self.c2(x)
        x = F.max_pool2d(F.relu(x),(2,2))
        x = self.c3(x)
        x = F.max_pool2d(F.relu(x),(2,2))
        if self.s == 1:
            self.s = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        self.f1 = nn.Linear(self.s,512)
        x = x.view(-1,self.s)
        x = F.relu(self.f1(x))
        x = F.log_softmax(self.f2(x),dim = -1)
        return x
    
      
        
model = Net() 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if (device.type=='cuda'):
    model.cuda() # CUDA

model.to(device)
for epoch in range(10):
    run_loss = 0.0
    for i, (data,target) in enumerate(train_loader):
        #target = target.squeeze(1)
        if (device.type=='cuda'):
            inputs,labels= Variable(data.cuda()), Variable(target.cuda())
        else:
            inputs,labels= Variable(data), Variable(target)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss =  F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        print('\r Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format( epoch, i * len(data), len(train_loader.dataset),100. * i / len(train_loader), loss.item()), end='')
    print(' ')
    
print('Finished')
output = model(testx)
pred = output.data.max(1)[1]
d = pred.eq(testy.data).cpu()
a=(d.sum().data.cpu().numpy())
b=d.size()
b=torch.tensor(b)
b=(b.sum().data.cpu().numpy())
accuracy = a/b
print('Accuracy:', accuracy*100)