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
#import required libraries

import torch

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torchvision import datasets, transforms,models

import helper

from torch.autograd import Variable

import torch.nn.functional as F

from torch.optim import lr_scheduler
print(os.listdir("../input/labelledrice/Labelled"))
#directories

directory="../input/labelledrice/Labelled"
#check if cuda is avaliable

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
#transforms

train_transforms=transforms.Compose([transforms.Resize(224),

                                    transforms.RandomResizedCrop(224224),

                                    transforms.RandomHorizontalFlip(p=0.2),

                                    transforms.RandomVerticalFlip(p=0.1),

                                     transforms.RandomRotation(30),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

                                    ])

valid_transforms=transforms.Compose([transforms.Resize(224),

                                    transforms.CenterCrop(224),

                                    transforms.ToTensor(),

                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
import torch.utils.data



#load data

image_datasets=dict()

image_datasets['train']=datasets.ImageFolder(directory,transform=train_transforms)

image_datasets['valid']=datasets.ImageFolder(directory,transform=valid_transforms)



#split dataset

num_workers=2

batch_size=128

valid_size=20.0





length_train=len(image_datasets['train'])

indices=list(range(length_train))

split = int(np.floor(valid_size * length_train))



train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)

valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)



#prepare data loaders



image_dataloaders=dict()

image_dataloaders["train"]=torch.utils.data.DataLoader(image_datasets["train"],batch_size=batch_size,sampler=train_sampler,shuffle=False)

image_dataloaders["valid"]=torch.utils.data.DataLoader(image_datasets["valid"],batch_size=batch_size,sampler=valid_sampler,shuffle=False)



image_dataloaders["valid"]
#view images
#build model

#retrained model

model=models.resnet152(pretrained=True)

for params in model.parameters():

    #freeze

    params.requre_grad=False
model

