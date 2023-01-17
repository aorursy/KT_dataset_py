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
from PIL import Image



from torchvision import datasets, transforms, models



import torch

from torch import nn, optim

from torch.optim import lr_scheduler

import torch.nn.functional as F
train_dir = "../input/train/"

test_dir = "../input/test/"
imgs = []



for f in os.listdir(train_dir):

    ext = os.path.splitext(f)[1]

    imgs.append(Image.open(os.path.join(train_dir,f)))
imgs[0]
class_df = pd.read_csv("../input/labels.csv")

class_df.head()
class_to_idx = {breed: index for index, breed in enumerate(class_df.breed.unique())}

class_to_idx
# Define transforms for the training and validation sets

train_transforms = transforms.Compose([transforms.RandomRotation(10),

                                transforms.RandomResizedCrop(224),

                                transforms.RandomHorizontalFlip(),

                                transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406], 

                                                     [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406], 

                                                     [0.229, 0.224, 0.225])])



# Load the datasets with ImageFolder

# !! not working for folders already separated with the training/testing data...

train_data = datasets.ImageFolder("../input/", transform = train_transforms)

test_data = datasets.ImageFolder("../input/", transform = test_transforms)



# Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)

testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = False)