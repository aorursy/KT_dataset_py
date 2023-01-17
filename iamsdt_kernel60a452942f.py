# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
# Imports

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from torch import nn, optim

import torch.nn.functional as F

import torchvision

from torchvision import datasets, transforms, models

from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler
!wget https://raw.githubusercontent.com/Iamsdt/DLProjects/master/utils/Helper.py
import Helper

import torch

from torchvision import datasets, transforms,models

from torch.utils.data import DataLoader



data_dir = '../input/labelledrice/Labelled/'



mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



train_transform = transforms.Compose([

                                transforms.RandomRotation(30),

                                transforms.Resize(255),

                                #transforms.RandomResizedCrop(224),

                                transforms.RandomHorizontalFlip(),

                                #transforms.ColorJitter(),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])

test_transform = transforms.Compose([

                                transforms.Resize(255),

                                #transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



train_data = datasets.ImageFolder(data_dir, transform=train_transform)

test_data = datasets.ImageFolder(data_dir, transform=test_transform)

print(len(train_data))



trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=32)



len(trainloader)
classes = train_data.classes
import numpy as np

import matplotlib.pyplot as plt



data_iter = iter(trainloader)

images, labels = data_iter.next() #this line



fig = plt.figure(figsize=(25, 5))

for idx in range(2):

    ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])

    # unnormolaize first

    img = images[idx] / 2 + 0.5

    npimg = img.numpy()

    img = np.transpose(npimg, (1, 2, 0)) #transpose

    ax.imshow(img, cmap='gray')

    ax.set_title(classes[labels[idx]])
model = models.resnet152(pretrained=True)

model.fc
model = Helper.freeze_parameters(model)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2048, out_features=1536),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=1536, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.3),

  nn.Linear(in_features=1024, out_features=4),

  nn.LogSoftmax(dim=1) 

)

    

model.fc = classifier

model.fc
import torch.optim as optim

import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
epoch = 1
model, train_loss, test_loss = Helper.train(model, trainloader, testloader, epoch, optimizer, criterion)
new_model = models.resnet50(pretrained=True)
m = nn.Sequential(*list(new_model.children())[:-1])

m
trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=16)
from tqdm import tqdm

# move to gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m.to(device)



#For training data



# Stores the labels of the train data

trn_labels = [] 



# Stores the pre convoluted features of the train data

trn_features = []



print("For training........")



#Iterate through the train data and store the calculated features and the labels

for data,label in tqdm(trainloader):

    o = m(Variable(data.to(device)))

    o = o.view(o.size(0),-1)

    trn_labels.extend(label)

    trn_features.extend(o.cpu().data)



#For test data

print("For testing........")



#Iterate through the validation data and store the calculated features and the labels

val_labels = []

val_features = []

for data,label in tqdm(testloader):

    o = m(Variable(data.to(device)))

    o = o.view(o.size(0),-1)

    val_labels.extend(label)

    val_features.extend(o.cpu().data)



print("Done")
from torch.utils.data import Dataset

class FeaturesDataset(Dataset):

    

    def __init__(self,featlst,labellst):

        self.featlst = featlst

        self.labellst = labellst

        

    def __getitem__(self,index):

        return (self.featlst[index],self.labellst[index])

    

    def __len__(self):

        return len(self.labellst)
#Creating dataset for train and validation

trn_feat_dset = FeaturesDataset(trn_features,trn_labels)

val_feat_dset = FeaturesDataset(val_features,val_labels)



#Creating data loader for train and validation

trn_feat_loader = DataLoader(trn_feat_dset,batch_size=1,shuffle=True)

val_feat_loader = DataLoader(val_feat_dset,batch_size=1)


classifier = nn.Sequential(

  nn.Linear(in_features=2048, out_features=1536),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=1536, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.3),

  nn.Linear(in_features=1024, out_features=4),

  nn.LogSoftmax(dim=1) 

)



net = classifier

print(net)
net.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.03)
epoch = 10
model, train_loss, test_loss = Helper.train(net, trn_feat_loader, val_feat_loader, epoch, optimizer, criterion)
Helper.test(model, val_feat_loader, criterion)