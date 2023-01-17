# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
!wget https://raw.githubusercontent.com/Iamsdt/DLProjects/master/utils/Helper.py
root_train = '../input/train_data/train_data'

root_test = '../input/test_data/test_data'
import Helper

import torch

from torchvision import datasets, transforms,models

from torch.utils.data import DataLoader



mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



train_transform = transforms.Compose([

                                transforms.Resize(255),

                                transforms.RandomResizedCrop(224),

                                transforms.RandomHorizontalFlip(),

                                transforms.ColorJitter(),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



test_transform = transforms.Compose([

                                transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



train_loader, test_loader, classes, class_to_idx = Helper.prepare_loader(

    root_train, root_test, train_transform, test_transform)



print("Total Class: ", len(classes))
Helper.visualize(test_loader, classes)
densenet = models.densenet161(pretrained=True)

densenet.classifier
densenet = Helper.freeze_parameters(densenet)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2208, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=1024, out_features=16),

  nn.LogSoftmax(dim=1)  

)

    

densenet.classifier = classifier

densenet.classifier
import torch.optim as optim

import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

densenet.to(device)



criterion = nn.NLLLoss()

optimizer = optim.Adam(densenet.classifier.parameters(), lr=0.003)

# turn this off

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
epoch = 5
densenet, train_loss, test_loss = Helper.train(densenet, train_loader, test_loader, epoch, optimizer, criterion)
Helper.check_overfitted(train_loss, test_loss)
resnet = models.resnet50(pretrained=True)

resnet.fc
resnet = Helper.freeze_parameters(resnet)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2048, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=1024, out_features=16),

  nn.LogSoftmax(dim=1)  

)

    

resnet.fc = classifier

resnet.fc
resnet.to(device)

optimizer = optim.Adam(resnet.fc.parameters(), lr=0.003)

# turn this off

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
resnet, train_loss, test_loss = Helper.train(resnet, train_loader, test_loader, epoch, optimizer, criterion)
Helper.check_overfitted(train_loss, test_loss)
incept = models.inception_v3(pretrained=True)



print(incept.fc)

incept.aux_logits = False

print(incept.aux_logits)
incept = Helper.freeze_parameters(incept)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2048, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=1024, out_features=16),

  nn.LogSoftmax(dim=1)  

)



classifier2 = nn.Sequential(

  nn.Linear(in_features=786, out_features=512),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=512, out_features=16),

  nn.LogSoftmax(dim=1)  

)

    

incept.fc = classifier

#incept.AuxLogits.fc = classifier2



print(incept.fc)

#print(incept.AuxLogits.fc)
incept.to(device)

optimizer = optim.Adam(incept.fc.parameters(),lr=0.003)

# turn this off

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
incept, train_loss, test_loss = Helper.train(incept, train_loader, test_loader, epoch, optimizer, criterion)
Helper.check_overfitted(train_loss, test_loss)
import torch.nn as nn

import torch





class MyEnsemble(nn.Module):



    def __init__(self, modelA, modelB, modelC, input):

        super(MyEnsemble, self).__init__()

        self.modelA = modelA

        self.modelB = modelB

        self.modelC = modelC



        self.fc1 = nn.Linear(input, 16)



    def forward(self, x):

        out1 = self.modelA(x)

        out2 = self.modelB(x)

        out3 = self.modelC(x)



        out = out1 + out2 + out3



        x = self.fc1(out)

        return torch.softmax(x, dim=1)
model = MyEnsemble(densenet, resnet, incept, 16)
model.to(device)

optimizer = optim.Adam(model.parameters(),lr=0.003)

# turn this off

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
model, train_loss, test_loss = Helper.train(model, train_loader, test_loader, epoch, optimizer, criterion)