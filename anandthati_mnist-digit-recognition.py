# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/train.csv')
Label = df['label']
Data = df.drop(['label'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(Data,Label,test_size=0.3)
class TrainDataset(Dataset):
    """ Train dataset."""

    # Initialize your data, download, etc.
    def __init__(self,X_Data, y_Data):
        self.len = X_Data.shape[0]
        self.x_data = torch.tensor(X_Data.values)
        self.y_data = torch.tensor(y_Data.values)
        print (self.x_data,self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = TrainDataset(X_train,y_train)
val_dataset = TrainDataset(X_test,y_test)

class TestDataset(Dataset):
    """ Test dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = pd.read_csv('../input/test.csv')
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy.values)

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


test_dataset = TestDataset()

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = x.reshape(in_size,1,28,28)
        x = x.type(torch.FloatTensor)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def val():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
final_out = []
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for image_id,data in enumerate(test_loader):
        data = Variable(data, volatile=True)
        output = model(data)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1].data.cpu().numpy() 
        final_out.append(pred)
    x = [j for i in final_out for j in i]
    return x
for epoch in range(1, 10):
    train(epoch)
    val()
result = test()
final_result=[y for x in result for y in x]
image_id = [i for i in range(1,len(final_result)+1)]
df_result = pd.DataFrame({'ImageId':image_id,'Label':final_result})
df_result.to_csv('final_submission.csv',index=None)
df_result
