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
!tar -zxvf ../input/cifar10-python/cifar-10-python.tar.gz
import torch 

import torchvision

import torchvision.transforms as transforms

#from ConvNet import ConvNet

import torch.nn as nn

# Convolutional neural network (two convolutional layers)

class ConvNet(nn.Module):

    def __init__(self, output_num, filer_size, strides, pad):

        super(ConvNet, self).__init__()

        #out_dim = int(1 + (input_num + 2 * pad - filer_size) / strides)

        self.layer1 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size = filer_size, stride= strides, padding=pad),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size = filer_size, stride = strides, padding = pad),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=0.05))

        self.layer3 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size = filer_size, stride = strides, padding = pad),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(4*4*128, 1024)

        self.Re = nn.ReLU()

        self.fc2 = nn.Linear(1024, 512)

        self.Re2 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(512, output_num)

        

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        out = self.Re(out)

        out = self.fc2(out)

        out = self.Re2(out)

        out = self.dropout(out)

        out = self.fc3(out)

        return out

def calculate_accuracy(loader, is_gpu):

    """Calculate accuracy.

    Args:

        loader (torch.utils.data.DataLoader): training / test set loader

        is_gpu (bool): whether to run on GPU

    Returns:

        tuple: (overall accuracy, class level accuracy)

    """

    correct = 0.

    total = 0.



    for data in loader:

        images, labels = data

        if is_gpu:

            images = images.to(device)

            labels = labels.to(device)



        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)



        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    print(100 * correct / total)

    return 100 * correct / total
# Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
# Hyper parameters

num_epochs = 50

num_classes = 10

batch_size = 100

learning_rate = 0.001

#input_num = 32 * 32

output_num = 10

filer_size = 3

hidden_unit =32

strides = 1

pad = 1
# CIFAR10 dataset

#load the data to tensor and scale the data into zero-mean, unit std

#trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) 



transform_train = transforms.Compose([

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261)),

])



# Normalize the test set same as training set without augmentation

transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),

])





train_dataset = torchvision.datasets.CIFAR10(root='.',

                                           train=True, 

                                           transform=transform_train,

                                           download=False)



test_dataset = torchvision.datasets.CIFAR10(root='.',

                                          train=False, 

                                          transform=transform_test)



# Data loader

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size=batch_size, 

                                           shuffle=True)



test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                          batch_size=batch_size, 

                                          shuffle=False)


model = ConvNet(output_num, filer_size, strides, pad).to(device)



# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

loss_list = []

acc_list = []

model.train()

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        # Run the forward pass

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss_list.append(loss.item())



        # Backprop and perform Adam optimisation

        optimizer.zero_grad()

        loss.backward()

        if(epoch>6):

            for group in optimizer.param_groups:

                for p in group['params']:

                    state = optimizer.state[p]

                    if(state['step']>=1024):

                        state['step'] = 1000

        optimizer.step()



        # Track the accuracy

        total = labels.size(0)

        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == labels).sum().item()

        acc_list.append(correct / total)



        if (i + 1) % 100 == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),

                          (correct / total) * 100))
model.eval()

print("Accuracy of test set", calculate_accuracy(test_loader, True))
