# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



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
!wget https://raw.githubusercontent.com/Iamsdt/60daysofudacity/master/day22/Helper.py
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



trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)



len(trainloader)
train_data.classes
classes = os.listdir(data_dir)

len(classes)
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
model = models.resnet50(pretrained=True)

model.fc
#model = Helper.freeze_parameters(model)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2048, out_features=4),

  #nn.ReLU(),

  #nn.Dropout(p=0.4),

  #nn.Linear(in_features=1536, out_features=1024),

  #nn.ReLU(),

  #nn.Dropout(p=0.3),

  #nn.Linear(in_features=1024, out_features=4),

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
epoch = 25
model, train_loss, test_loss = Helper.train(model, trainloader, testloader, epoch, optimizer, criterion)
model = Helper.load_latest_model(model)
Helper.check_overfitted(train_loss, test_loss)
Helper.test(model, testloader, criterion)
Helper.test_per_class(model, testloader, criterion, classes)