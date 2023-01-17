# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

import numpy as np

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models
data_dir = '/kaggle/input/flowers-recognition/'

batch_size = 16

# Defineing transforms for the training data and testing data

train_transforms = transforms.Compose([transforms.Resize((224,224)),

                                       transforms.RandomRotation(20),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.4, 0.4, 0.4],

                                                           [0.3, 0.3, 0.3])])



test_transforms = transforms.Compose([transforms.Resize((224,224)),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.4, 0.4, 0.4],

                                                           [0.3, 0.3, 0.3])])







test_transforms = transforms.Compose([transforms.Resize((224,224)),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.4, 0.4, 0.4],

                                                           [0.3, 0.3, 0.3])])

train_data = datasets.ImageFolder(data_dir, transform=train_transforms)





from torchvision import datasets

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
# number of subprocesses to use for data loading

num_workers = 0

# how many samples per batch to load

batch_size = 20

# percentage of training set to use as validation

valid_size = 0.2

test_size = 0.2



# obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

split2 = int(np.floor(test_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,

    sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

    sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

    num_workers=num_workers)
