# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy# linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Create the Test and train Dataloaders 

from torch.utils.data import DataLoader

from torchvision import transforms

import torchvision

import torch

import torch.nn as nn



train_path = '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'



torch.manual_seed(123)

train_data_loader = DataLoader(

    torchvision.datasets.ImageFolder(

        train_path,

        transform=transforms.Compose([

            transforms.Resize((224, 224)),  # INPUT IMAGE SIZE FOR SQUEEZENET

            transforms.RandomHorizontalFlip(),

            transforms.RandomAffine(degrees=10),

            transforms.RandomPerspective(),

            transforms.RandomRotation(degrees=15),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    ),

    num_workers=8,

    batch_size=256,

    shuffle=False,

    pin_memory=True

)

#Create the Architecture Class. Squeezenet achieves Alexnet Level accuracy with very less Parameters. 

#Do checkout my github Repo containing the Code - https://github.com/AD2605/Squeezenet-Pytorch



class Fire(nn.Module):

    def __init__(self,in_channels,squeeze_channels,k1_channels,k3_channels):

        super(Fire, self).__init__()



        self.in_channels = in_channels

        self.squeeze_channels = squeeze_channels

        self.k1_channels = k1_channels

        self.k3_channels = k3_channels



        self.squeeze_layer = self.get_squeeze_layer().cuda()

        self.expand1_layer = self.expand_1_layer().cuda()

        self.expand3_layer = self.expand_3_layer().cuda()



    def get_squeeze_layer(self):

        layers = []

        layers.append(nn.Conv2d(self.in_channels,self.squeeze_channels,kernel_size=1))

        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)



    def expand_1_layer(self):

        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,self.k1_channels,kernel_size=1))

        layers.append(nn.ReLU(inplace=True))



        return nn.Sequential(*layers)



    def expand_3_layer(self):

        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,self.k3_channels,kernel_size=3,padding=1))

        layers.append(nn.ReLU(inplace=True))



        return nn.Sequential(*layers)



    def forward(self, x):

        y = self.squeeze_layer(x)

        return torch.cat([self.expand1_layer(y),self.expand3_layer(y)], dim=1)



class SqueezeNet(nn.Module):

    def __init__(self,channels,classes ):

        super(SqueezeNet, self).__init__()



        self.channels = channels

        self.classes = classes



        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=3, stride=2)

        self.layers = []

        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        self.layers.append(Fire(64, 16, 64, 64))

        self.layers.append(Fire(128, 16, 64, 64))

        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        self.layers.append(Fire(128, 32, 128, 128))

        self.layers.append(Fire(256, 32, 128, 128))

        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        self.layers.append(Fire(256, 48, 192, 192))

        self.layers.append(Fire(384, 48, 192, 192))

        self.layers.append(Fire(384, 64, 256, 256))

        self.layers.append(Fire(512, 64, 256, 256))

        self.layers.append(nn.Dropout())

        self.layers.append(nn.Conv2d(512, self.classes, kernel_size=1))

        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.AvgPool2d(13, stride=1))

        for layer in self.layers:

            layer.cuda()



    def forward(self, x):

        out = self.conv1(x)

        for layer in self.layers:

            out = layer(out)

        if self.classes ==2:

            return out.view(out.size(0), 1)

        else:

            return out.view(out.size(0), self.classes)



    def train_model(self, model, data, epochs):

        if self.classes ==2:

            criterion = nn.BCEWithLogitsLoss().cuda()

        else:

            criterion = nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10)

        min_loss = 5000

        model.train()

        model.cuda()

        for epoch in range(0, epochs):

            train_accuracy = 0

            net_loss = 0

            for _, (x, y) in enumerate(data):

                optimizer.zero_grad()

                x = x.cuda()

                y = y.cuda()

                out = model(x)

                loss = criterion(out, y)

                loss.backward()

                optimizer.step()

                max_index = out.max(dim=1)[1]

                accuracy = (max_index==y).sum()

                train_accuracy += accuracy.item()

                net_loss+=loss.item()

            scheduler.step()

            print('---------------------------------------------------------')

            print(epoch)

            print('AVERAGE LOSS = ', net_loss/len(data))

            print('TRAIN ACCURACY = ', train_accuracy/len(data))

            scheduler.step()

            if net_loss<min_loss:

                torch.save(model.state_dict(), '/kaggle/working/squeezenet.pth')



    def evaluate(self, model, dataloader):

        model.eval()

        for parameter in model.parameters():

            parameter.requires_grad = False

        correct = 0

        model.cuda()

        for _, (x, y) in enumerate(dataloader):

            x = x.cuda()

            y = y.cuda()

            out = model(x)

            if torch.argmax(out) == y:

                correct += 1

        print(correct / len(dataloader))
#Call the model class and train. Training with a batch size of 256. 

squeezeNet = SqueezeNet(channels=3, classes=29).cuda()

squeezeNet.train_model(epochs=50, data=train_data_loader, model=squeezeNet)