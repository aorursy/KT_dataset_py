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
import torch.nn as nn

import torch 

import torchvision

import torchvision.transforms as transforms



import numpy as np 

import matplotlib.pyplot as plt

batch_size_train = 1000

batch_size_test = 1000

num_repeat_list = [2, 4, 4, 2]

num_strides_list = [1, 2, 2, 2]

num_epochs = 50
# 3x3 convolutional layer, padding for all the conv3x3 is 1

def conv3x3(in_channels, out_channels, stride):

    conv3x3 = nn.Conv2d(in_channels, 

                    out_channels,

                    kernel_size = 3,

                    stride = stride,

                    padding = 1,

                    bias=False)

    return conv3x3
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):

        super().__init__()

        

        self.conv = conv3x3(in_channels, out_channels, stride)

        

        self.BN = nn.BatchNorm2d(out_channels)

        

        self.ReLu = nn.ReLU()

        

        self.conv2 = conv3x3(out_channels, out_channels, stride = 1)

        

        self.BN2 = nn.BatchNorm2d(out_channels)

        

        self.downsample = downsample

        

        self.stride = stride

        

    def forward(self, x):

        residual = x

        #print("x", x.shape)

        out = self.conv(x)

        #print("out", out.shape)

        

        out = self.BN(out)

        #print("out", out.shape)

        

        out = self.ReLu(out)

        #print("out", out.shape)

        

        out = self.conv2(out)

        #print("out", out.shape)

        

        out = self.BN2(out)

        #print("BN2 out", out.shape)

        #print(self.downsample)

        if self.downsample is not None:

            residual = self.downsample(x)

        

        #print("out dim", out.shape, "residual dim", residual.shape)

        out += residual

        return out

    
class ResNet(nn.Module):

    def __init__(self, in_channels, num_class, num_repeat_list, num_strides_list):

        super().__init__()

        

        self.conv = conv3x3(in_channels, 32, stride = num_strides_list[0])

        self.BN = nn.BatchNorm2d(32)

        self.ReLu = nn.ReLU()

        self.dropout = nn.Dropout2d(p = 0.1) # dropout prob

        self.in_channels = 32

        self.conv2_x = self._build_block(BasicBlock,

                                    32,

                                    num_repeat_list[0],

                                    num_strides_list[0])

        

        self.conv3_x = self._build_block(BasicBlock,

                                    64,

                                    num_repeat_list[1],

                                    num_strides_list[1])        

        

        self.conv4_x = self._build_block(BasicBlock,

                                    128,

                                    num_repeat_list[2],

                                    num_strides_list[2])        

        

        self.conv5_x = self._build_block(BasicBlock,

                                    256,

                                    num_repeat_list[3],

                                    num_strides_list[3])

        

        self.max_pooling = nn.MaxPool2d(kernel_size = 2,

                                        stride =2)

        self.fc = nn.Linear(1024, num_class) # input = ?, output class = num_class

            #def initialize_wieght(self, layers)

        #init weight

    """

    

    def _build_block(self, block,  out_channels, duplicates, stride=1):

        downsample = None

        if (stride != 1) or (self.in_channels != out_channels):

            downsample = nn.Sequential(

                conv3x3(self.in_channels, out_channels, stride=stride),

                nn.BatchNorm2d(num_features=out_channels)

            )



        layers = []

        print("self.in_channels", self.in_channels)

        

        layers.append(

            block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels

        print("updated self.in_channels", self.in_channels)



        for _ in range(1, duplicates):

            layers.append(block(out_channels, out_channels))



        return nn.Sequential(*layers)

    

    """

    def _build_block(self, basic_block, out_channels, repeat = 0, stride = 1):

        #create downsampling block

        downsample = None

        if self.in_channels != out_channels:

            downsample = nn.Sequential(conv3x3(self.in_channels,

                                               out_channels,

                                               stride=stride),

                                    nn.BatchNorm2d(out_channels))

        #initial the list

        layers =[]

        #create a adaptive layer

        layers.append(basic_block(self.in_channels, out_channels, stride, downsample))

        

        #update the channel for the rest of the blocks

        self.in_channels = out_channels

        

        for i in range(1, repeat):

            layers.append(basic_block(self.in_channels, out_channels, 1))

        

        return nn.Sequential(*layers)

    

    

    def forward(self, x):

        out = self.conv(x)

        out = self.BN(out)

        out = self.ReLu(out)

        

        #basic block

        out = self.dropout(out)

        out = self.conv2_x(out) 

        out = self.conv3_x(out)

        out = self.conv4_x(out)

        out = self.conv5_x(out)

        out = self.max_pooling(out)

                

        #flatten

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out

    

# Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
#define data transform 



transform_train = transforms.Compose([

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761)),

])



# Normalize the test set same as training set without augmentation

transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),

])





#load data to dataloader

trainset = torchvision.datasets.CIFAR100(root='/kaggle/input/cifar100/cifar-100-python/',

                                         train=True,

                                         download=False,

                                         transform=transform_train)

train_loader = torch.utils.data.DataLoader(trainset,

                                          batch_size=batch_size_train,

                                          shuffle=True)



testset = torchvision.datasets.CIFAR100(root='/kaggle/input/cifar100/cifar-100-python/',

                                        train=False,

                                        download=True,

                                        transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset,

                                         batch_size=batch_size_test,

                                         shuffle=False)

#class ResNet(nn.Module):

#    def __init__(self, in_channels, num_class, num_repeat_list, num_strides_list):

#

model = ResNet(3, 100, num_repeat_list, num_strides_list).to(device)



# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2, weight_decay=1e-2)
print(model)
total_step = len(train_loader)

loss_list = []

acc_list = []

model.train()

for epoch in range(num_epochs):

    running_loss = 0

    for i, (images, labels) in enumerate(train_loader):

        # Run the forward pass

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        #loss_list.append(loss.item())



        # Backprop and perform Adam optimisation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        # Track the accuracy

        total = labels.size(0)

        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == labels).sum().item()

        acc_list.append(correct / total)



        if (i) % 100 == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),

                      (correct / total) * 100))
