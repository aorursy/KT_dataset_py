import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import os

print(os.listdir("../input/fruits-360_dataset/fruits-360"))



import torch

import torchvision

from torchvision import transforms

from torch.utils.data import TensorDataset,DataLoader





transformation = transforms.Compose([

        transforms.Resize(size=(200,200)),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0,0,0), std=(1,1,1))])



batch_size = 64

train_dataset = torchvision.datasets.ImageFolder("../input/fruits-360_dataset/fruits-360/Training", transform=transformation)

train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

test_ds = torchvision.datasets.ImageFolder("../input/fruits-360_dataset/fruits-360/Test", transform=transformation)

test_loader = DataLoader(test_ds,batch_size=batch_size, shuffle=True)
import torch.nn as nn

import torch.nn.functional as F

import math

class NN(nn.Module):

    def __init__(self):

        super(NN, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=50, kernel_size=4),

            nn.ReLU(),

            nn.MaxPool2d(3),

            nn.Conv2d(in_channels=50, out_channels=64, kernel_size=3),

            nn.ReLU(),

            nn.MaxPool2d(3),

            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3),

            nn.ReLU(),

            nn.MaxPool2d(5))

        

            

        self.layer2 = nn.Sequential(

            nn.Linear(720,128),

            nn.ReLU(),

            nn.Linear(128,128),

            nn.ReLU(),

            nn.Linear(128,103)

        )

    def forward(self, x):

        y = self.layer2(self.layer1(x).view(x.size(0), -1))

        return y
def fit(model, train_dl, lr=0.005, epoches=5):

    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None

    best_acc = 0

    for epoche in range(epoches):

        ep_loss = 0

        for xx,yy in train_dl:

            xx,yy = xx.cuda(), yy.cuda()

            optimizer.zero_grad()

            y_pred = model(xx)

            loss = criterion(y_pred, yy)

            loss.backward()

            ep_loss+=loss.item()

            optimizer.step()

        print("Loss: {}".format(ep_loss/len(train_dl)))

    model.cpu()
net = NN()

net.cpu()

fit(net,train_loader,epoches=20)
y_true = []

y_pred = []

for xx,yy in test_loader:

    net.cuda()

    xx,yy = xx.cuda(), yy.cuda()

    out = net(xx).argmax(dim=1)

    y_true.extend(yy.tolist())

    y_pred.extend(out.tolist())
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))

print(confusion_matrix(y_true, y_pred))