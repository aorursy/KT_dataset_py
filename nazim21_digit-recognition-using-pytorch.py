import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as f

import torch.optim as optim

from torch.autograd import Variable

import matplotlib.pyplot as plt

from torch.utils.data import Dataset

%matplotlib inline 



import os

print(os.listdir("../input"))
#hyperparameters

num_epochs = 20

batch_size = 128

num_classes = 10

learning_rate = 0.001
class loadMNIST(Dataset):

    def __init__(self, type,transforms=None):

        df = pd.read_csv('../input/'+type+'.csv')

        self.len = df.shape[0]

        self.type = type

        self.transforms = transforms

        if type=='train':

            self.X_data = torch.from_numpy(df.iloc[:,1:].values.reshape(self.len,28,28))

            self.y_data = torch.from_numpy(df.iloc[:,:1].values.reshape(self.len,1))

        else:

            self.X_data = torch.from_numpy(df.values.reshape(self.len,28,28))

    

    def __getitem__(self,index):

        if self.transforms is not None:

            self.X_data = self.transforms(self.X_data)

            

        if self.type == 'train':

            return (self.X_data[index], self.y_data[index])

        else:

            return (self.X_data[index])

    def __len__(self):

        return self.len

        
transform = transforms.Compose([transforms.Normalize((0.5,),(0.5,))])

trainset = loadMNIST('train', transform)

testset = loadMNIST('test', transform)
data = torch.utils.data.random_split(trainset, [40000, 2000])
data
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=1)

testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=True, num_workers=1)
class LeNet5(nn.Module):

    def __init__(self):

        super(LeNet5, self).__init__()



        #1 input image channel, 6 output channels, 5x5 square convolution kernel

        self.conv1 = nn.Conv2d(1,6,5)

        self.conv1_bn = nn.BatchNorm2d(6)

        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(6,16,5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16*4*4,120)

        self.fc1_bn = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(120,84)

        self.fc2_bn = nn.BatchNorm1d(84)

        self.fc3 = nn.Linear(84,10)

    

    def forward(self,x):

        x = self.pool(self.conv1_bn(f.relu(self.conv1(x))))

        x = self.pool(self.conv2_bn(f.relu(self.conv2(x))))

        

        # If the size is a square you can only specify a single number

        x = x.view(-1,16*4*4)

        

        x = self.fc1_bn(f.relu(self.fc1(x)))

        x = self.fc2_bn(f.relu(self.fc2(x)))

        x = self.fc3(x)

        return f.softmax(x)



net = LeNet5()

print(net)
def init_weights(m):

    if isinstance(m, nn.Conv2d):

        nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:

            nn.init.zeros_(m.bias)



net.apply(init_weights)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), 0.001)

if torch.cuda.is_available():

    criterion = criterion.cuda()
train_acc = []

test_acc = []

for epoch in range(num_epochs):

    if torch.cuda.is_available():

        net = net.cuda()

    for (inputs, labels) in trainloader:

        print(inputs.shape)

        optimizer.zero_grad()

        

        if torch.cuda.is_available():

            inputs, labels = inputs.cuda(),labels.cuda()

            

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    

    train_correct = 0

    train_total = 0

    net = net.cpu()

    with torch.no_grad():

        for (images, labels) in trainloader:

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)

            train_correct += (predicted == labels).sum().item()

            

    train_accuracy = (100*train_correct)/(train_total)

    train_acc.append(train_accuracy)

    

    correct = 0

    total = 0

    net = net.cpu()

    with torch.no_grad():

        for (images, labels) in valloader:

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



    test_accuracy = (100*correct)/(total)

    test_acc.append(test_accuracy)

    

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%,  Test Accuracy: {test_accuracy:.2f}%")    
correct = 0

total = 0

net = net.cpu()

with torch.no_grad():

    for (images, labels) in testloader:

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print(f'Final Accuracy of the network on test images is: {(correct/total)*100:.2f}%')