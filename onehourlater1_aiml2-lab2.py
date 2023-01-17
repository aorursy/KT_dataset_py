import numpy as np

import matplotlib.pyplot as plt

import random

import os

import time

from sklearn.metrics import classification_report, confusion_matrix



import numpy as np 

import pandas as pd
import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
def get_classes():

    return sorted(os.listdir('../input/fruits-360_dataset/fruits-360/Training'))

    

classes = get_classes()

print(classes)

img_size = 100

img_sc = int(((img_size-2)/4)-1)
class Conv(nn.Module):

    def __init__(self, layer1_neurons, layer2_neurons):

        super(Conv, self).__init__()

        

        self.layer1_neurons = layer1_neurons

        self.layer2_neurons = layer2_neurons

        

        self.pool = nn.MaxPool2d(2, 2)

        

        self.img_sc_param = (img_sc**2)*20

        

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 100, kernel_size = 3)

        self.conv2 = nn.Conv2d(in_channels = 100, out_channels = 20, kernel_size = 3)

        

        self.lin1 = nn.Linear(self.img_sc_param, self.layer1_neurons)

        self.lin2 = nn.Linear(self.layer1_neurons, self.layer2_neurons)

        self.lin3 = nn.Linear(self.layer2_neurons, 104)



    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)

        x = self.pool(x)



        x = self.conv2(x)

        x = F.relu(x)

        x = self.pool(x)

        

        x = x.view(-1, self.img_sc_param)

        

        x = self.lin1(x)

        x = F.relu(x)

        

        x = self.lin2(x)

        x = F.relu(x)

        

        x = self.lin3(x)

        

        return x



def train(conv, loader, epochs):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(conv.parameters(), lr = 0.01, momentum = 0.9)

    

    for epoch in range(epochs):

        r_loss = 0.0

        for i, data in enumerate(loader, 0):

            inputs, labels = data



            optimizer.zero_grad()



            outputs = conv(inputs.cuda())

            loss = criterion(outputs, labels.cuda())

            loss.backward()

            optimizer.step()



            r_loss += loss.item()

            

            every_n = 700

            if i % every_n == every_n - 1:

                print('[%d, %5d] loss: %.3f' %

                      (epoch + 1, i + 1, r_loss / every_n))

                r_loss = 0.0



def test(conv, test_loader):

    #print(len(test_loader))

    dataiter = iter(test_loader)

    classes = get_classes()

    

    #print(len(dataiter))

    

    for i in range(10000):

        images, labels = dataiter.next()

        #print(labels)

        if i % 1000==0:

            outputs = conv(images.cuda())

            _, predicteds = torch.max(outputs, 1)

            predicted = classes[predicteds[0]]

            groundtruth = classes[labels[0]]

            

            #print(i,"-",predicteds[0])

            print(i)

            print("Prediction: {}".format(predicted))

            print("Real: {}".format(groundtruth))

        



def get_predictions(conv, data_loader):

    y_true = []

    y_pred = []

    

    for data in data_loader:

        images, labels = data

        images = images.cuda()

        labels = labels.cuda()

        

        outputs = conv(images)

        _, predicted = torch.max(outputs.data, 1)

        y_true += labels.tolist()

        y_pred += predicted.tolist()

    

    return [y_true, y_pred]





def validate(conv, test_loader):

    y_true, y_pred = get_predictions(conv, test_loader)

    

    correct = (np.array(y_true) == np.array(y_pred)).sum()

    total = len(y_true)

    validation = correct / total

    

    return validation



def get_metrics(conv, test_loader):

    y_true, y_pred = get_predictions(conv, test_loader)

    

    classes = get_classes()



    y_true = [classes[a] for a in y_true]

    y_pred = [classes[a] for a in y_pred]

    

    classification = classification_report(y_true, y_pred, labels = classes, target_names = classes)

    

    confusion = confusion_matrix(y_true, y_pred, labels = classes)

    

    return [classification, confusion]

conv = Conv(3000, 300)



conv.cuda()
transform = transforms.Compose([

    transforms.Resize(size = (img_size, img_size)),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0,0,0), std=(1,1,1))

])



train_data = torchvision.datasets.ImageFolder('../input/fruits-360_dataset/fruits-360/Training', transform = transform)

test_data = torchvision.datasets.ImageFolder('../input/fruits-360_dataset/fruits-360/Test', transform = transform)
train_data_split, validation_data_split = torch.utils.data.random_split(train_data, [int(len(train_data)*0.4), len(train_data)-int(len(train_data)*0.4)])



train_loader = torch.utils.data.DataLoader(train_data_split, 10, shuffle=True, num_workers = 2)

validation_loader = torch.utils.data.DataLoader(validation_data_split, 1, shuffle=True, num_workers = 2)

test_loader = torch.utils.data.DataLoader(test_data, 1, shuffle=True, num_workers = 2)

print("Train")

train(conv, train_loader, 5)



print('Validation')

conv_validation = validate(conv, validation_loader)



print('Validation score')

print(conv_validation)
print("Test")

test(conv, test_loader)
classification, confusion = get_metrics(conv, test_loader)

print()

print('classification report:')

print(classification)



print()

print('confusion report:')

print(confusion)