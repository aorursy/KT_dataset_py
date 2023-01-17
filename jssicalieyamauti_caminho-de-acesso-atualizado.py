from torchvision import transforms, datasets, models

import torch

from torch import optim, cuda

from torch.utils.data import DataLoader, sampler, random_split

import torch.nn as nn



from PIL import Image

import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
print(os.listdir('../input/fotos-ia'))
image_transforms = {

    # Train uses data augmentation

    'train':

    transforms.Compose([

        transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=224),  # Image net standards

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])  # Imagenet standards

    ]),

    'val':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
batch_size = 128



all_data = datasets.ImageFolder(root='../input/fotos-ia')

train_data_len = int(len(all_data)*0.8)

valid_data_len = int((len(all_data) - train_data_len)/2)

test_data_len = int(len(all_data) - train_data_len - valid_data_len)

train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])

train_data.dataset.transform = image_transforms['train']

val_data.dataset.transform = image_transforms['val']

test_data.dataset.transform = image_transforms['test']

print(len(train_data), len(val_data), len(test_data))



trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  #canais de entrada (camada), canais de saida (filtros), tamanho da janela (filtro)

        self.pool = nn.MaxPool2d(2, 2)  #(tamanho da area, deslocamento) sumarização

        self.conv2 = nn.Conv2d(6, 16, 5)  

        self.conv2_bn = nn.BatchNorm2d(16)  #quantidade da camada anterior

        self.fc1 = nn.Linear(16 * 53 * 53, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 2)

        

    def forward(self, x):

        x = F.relu(self.conv1(x)) #separa valores maior de zero e menor que zero

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        print(x.shape)

        x = x.view(-1, 16 * 53 * 53)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



net = Net()

net.to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 2000))

            running_loss = 0.0



print('Finished Training')
correct = 0

total = 0



with torch.no_grad():

    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))