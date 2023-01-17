import torch

import torch.nn as nn

import torch.nn.functional as f

import torch.utils.data as data_utils

from torch.utils.data.dataset import Dataset

import torchvision

import torchvision.transforms as transforms

from torch.utils import data

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import os



for dirname, _, filenames in os.walk('/kaggle/input/pretrainednet/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#define transformations

transforms_train = transforms.Compose([transforms.Resize(250),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.RandomRotation(12),

                                       transforms.CenterCrop(224), 

                                       transforms.ToTensor(), 

                                       transforms.Normalize((0.485, 0.456, 0.406), 

                                                            (0.229, 0.224, 0.225))])



transforms_test = transforms.Compose([transforms.Resize(250), 

                                      transforms.CenterCrop(224), 

                                      transforms.ToTensor(), 

                                      transforms.Normalize((0.485, 0.456, 0.406), 

                                                           (0.229, 0.224, 0.225))])



#load training and test set

train_set = torchvision.datasets.ImageFolder(

                            root="/kaggle/input/cat-and-dog/training_set/training_set/", 

                            transform=transforms_train)



train_loader = data.DataLoader(train_set, batch_size=100, shuffle=True)

     

test_set = torchvision.datasets.ImageFolder(

                            root="/kaggle/input/cat-and-dog/test_set/test_set/", 

                            transform=transforms_test)



test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True)
#data seems to be balanced

print(np.bincount(train_set.targets))

print(train_set.classes)

print(train_set.class_to_idx)
#define model and load weights

model = torchvision.models.resnet18()

state_dict = torch.load("/kaggle/input/pretrainednet/resnet18-5c106cde.pth")

model.load_state_dict(state_dict)

print(model)
#freeze parameters

for param in model.parameters(): 

    param.requires_grad = False
#cut of the classifier/last layer of the model and add new ones

model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512),

                         nn.BatchNorm1d(512),

                         nn.ReLU(True),

                         nn.Dropout(p=0.2),

                         nn.Linear(512, 2),

                         nn.Softmax(dim=1))
#loss functioin and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#train and validation

for epoch in range(1):

    model.train() 

    acc_train = []

    acc_test = []

        

    for X,y in train_loader:

        optimizer.zero_grad()

        out = model(X)

        

        pred = out.detach().numpy()

        

        label = y.detach().numpy()

        a = (pred.argmax(axis=1) == label)

        acc_train.extend(a)

        

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        

      

    print("Training Accuracy for {}: {}%".format(epoch+1, sum(acc_train) / len(acc_train) * 100))

        

    model.eval()

    with torch.no_grad():

        for X, y in test_loader:

            out = model(X)

            

            pred = out.detach().numpy()

            label = y.detach().numpy()

            a = (pred.argmax(axis=1) == label)

            acc_test.extend(a)       

            loss = criterion(out, y)

            

        print("Validation Accuracy for {}: {}%".format(epoch+1, sum(acc_test) / len(acc_test) * 100))