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
from math import *
from matplotlib.pyplot import imshow
%matplotlib inline
import matplotlib.pyplot as plt
file_path = "../input/fashion-mnist_train.csv"
traindata  = pd.read_csv(file_path)
batch_size = 10
data_length =  len(traindata)
batch_num = floor( data_length/batch_size )
test_file_path = "../input/fashion-mnist_test.csv"
testdata  = pd.read_csv(test_file_path)
test_batch_size = 10
test_data_length =  len(testdata)
test_batch_num = floor( test_data_length/test_batch_size )
index = 0
train_x = traindata[ traindata.columns[1:] ].iloc[index * batch_size: (index + 1)* batch_size].values
sample_x = train_x.reshape(10, 28, 28)[5]
imshow(sample_x, cmap='gray')
def getData(index):
    train_y = traindata[ traindata.columns[0] ].iloc[index * batch_size: (index + 1)* batch_size].values
    train_x = traindata[ traindata.columns[1:] ].iloc[index * batch_size: (index + 1)* batch_size].values
    train_x_s =  np.reshape( train_x, ( batch_size, 1, 28, 28 ) )
    tensor_y = torch.from_numpy(train_y)
    tensor_x = torch.from_numpy(train_x_s).float()
    return tensor_x, tensor_y
def gettestData(index):
    test_y = testdata[ testdata.columns[0] ].iloc[index * test_batch_size: (index + 1)* test_batch_size].values
    test_x = testdata[ testdata.columns[1:] ].iloc[index * test_batch_size: (index + 1)* test_batch_size].values
    test_x_s =  np.reshape( test_x, ( test_batch_size, 1, 28, 28 ) )
    tensor_y = torch.from_numpy(test_y)
    tensor_x = torch.from_numpy(test_x_s).float()
    return tensor_x, tensor_y
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
lossdata = ([],[])
for index in range(10):
    optimizer.zero_grad()
    train_x, train_y = getData(index)
    output = net.forward(train_x)
    loss = criterion(output, train_y)
    lossdata[0].append(index)
    lossdata[1].append(loss)
    loss.backward()
    optimizer.step()
    print( "loss at {}: {}".format( index ,loss) )
line,  = plt.plot(lossdata[0], lossdata[1], "-", linewidth=2)
torch.argmax(net.forward(gettestData(0)[0]), dim=1)
gettestData(0)[1]
net.forward(gettestData(0)[0])[0]
(gettestData(0)[1] == torch.argmax(net.forward(gettestData(0)[0]), dim=1)).sum().item()
cumCorrect = 0
for index in range(10):
    test_x, test_y = gettestData(index)
    output = net.forward(test_x)
    argmax = torch.argmax(output, dim=1)
    correctNum = (argmax == test_y).sum().item()
    cumCorrect += correctNum
    
    print( "{} th correct {} percent ".format(index, correctNum/10*100) )
print("test accuracy: {}".format(cumCorrect/10000 * 100) )
