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
import torch
import torchvision
from torchvision import datasets,transforms
train = datasets.MNIST("",train = True , download = True , transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("",train = False , download = True , transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size = 10 , shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size = 10 , shuffle = True)

for data in trainset :
    print(data)
    break
x,y = data[0][0] , data[1][0]

import matplotlib.pyplot as plt
plt.imshow(data[0][9].view(28,28))
plt.show()
#import torch.nn as nn
#import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,10)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim =1)
    

X = torch.rand((28,28))
X = X.view(-1,28*28)
net = Net()
net(X)


import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

for epoch in range(3): #3 full pass over the data
    for data in trainset: #data in batch data
        X,y = data #X is the batch features and y is batch of targets
        net.zero_grad() #set gradient to zero before loss calc.you will do it for each step
        output = net(X.view(-1,28*28)) #pass in the reshaped batch
        loss = F.nll_loss(output,y) #cal and store the loss
        loss.backward() #apply this loss backwards thru the networks parameters
        optimizer.step() #attempt to optmize weights to account for loss/gradients
    print(loss)
correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X,y = data
        output = net(X.view(-1,28*28))
        for idx , i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct = correct +1
            total = total +1

print("Accuracy",round(correct/total,3))        
plt.imshow(X[9].view(28,28))
print(torch.argmax(net(X[9].view(-1,28*28))[0]))