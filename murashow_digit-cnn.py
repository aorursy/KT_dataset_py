# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dirname
pd.read_csv(os.path.join(dirname, "sample_submission.csv"))
train = pd.read_csv(os.path.join(dirname, "train.csv"))
test = pd.read_csv(os.path.join(dirname, "test.csv"))
train_label=train["label"].values
train_data=train.drop(["label"],axis=1).values.reshape(len(train),1,28,28)
test_data=test.values.reshape(len(test),1,28,28)
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

trainset = TensorDataset(torch.from_numpy(train_data),
                        torch.from_numpy(train_label))
testset = TensorDataset(torch.from_numpy(test_data),
                       torch.from_numpy(np.zeros(len(test_data))))
BATCH_SIZE = 100

trainloader = DataLoader(trainset,batch_size=BATCH_SIZE, shuffle=True, num_workers=2 )
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1, 16, 3)
        self.conv2=nn.Conv2d(16,32, 3)
        self.batch1=nn.BatchNorm2d(16)
        self.batch2=nn.BatchNorm2d(32)
        self.pool=nn.MaxPool2d(2,stride=2)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(32*5*5,120)
        self.fc2=nn.Linear(120,10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.pool(x)
        x = x.view(len(x),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
net = Net()
device = torch.device('cuda')
net.to(device)
criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.0001,weight_decay=0.005)

EPOCH=10
predictions = []
for epoch in range(EPOCH):
    for inputs, labels in trainloader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        output = net(inputs.float())
        loss = criterion(output,labels.long())
        loss.backward()
        optimizer.step()
        
for inputs, labels in testloader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs.float())
        _, p = outputs.max(1) 
        for i in p:
            predictions.append(i.item())
#         loss = criterion(output,labels.long())
        
        
pd.DataFrame(np.stack([np.array([ i+1 for i in range(len(predictions))]),np.array(predictions).reshape(-1)]).T, columns=["imageId", "Label"]).to_csv("submission.csv", index=False)