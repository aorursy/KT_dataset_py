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

data_dir = '../input/dataset_itr2/dataset_itr2'

di = os.listdir(data_dir+'/train')[5]

name = os.listdir(data_dir+'/train/'+di)[5]

print("Label: ",di)

Image.open(data_dir+"/train/"+di+"/"+name)
!wget https://raw.githubusercontent.com/Iamsdt/60daysofudacity/master/day22/Helper.py
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



train_data  = datasets.ImageFolder(data_dir + '/train', train_transform)

test_data = datasets.ImageFolder(data_dir + '/test', test_transform)
classes = train_data.classes

class_idx = train_data.class_to_idx

class_idx
len(classes)
train_loader = torch.utils.data.DataLoader(train_data,

                                           batch_size=64,

                                           shuffle=True,

                                           num_workers=1)



test_loader = torch.utils.data.DataLoader(test_data,

                                          batch_size=64,

                                          shuffle=False,

                                          num_workers=1)



print(len(train_loader))
import Helper

Helper.visualize(train_loader, classes)
model = models.densenet161(pretrained=True)

model.classifier
model = Helper.freeze_parameters(model)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2208, out_features=1536),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=1536, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.3),

  nn.Linear(in_features=1024, out_features=35),

  nn.LogSoftmax(dim=1)  

)

    

model.classifier = classifier

model.classifier
import torch.optim as optim

import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)



criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
epoch = 2
model, train_loss, test_loss = Helper.train(model, train_loader, test_loader, epoch, optimizer, criterion)
model = Helper.load_latest_model(model)
Helper.check_overfitted(train_loss, test_loss)
Helper.test(model, test_loader)
Helper.test_per_class(model, test_loader, criterion, classes)
from PIL import Image



def test(file):

  ids = train_loader.dataset.class_to_idx



  with Image.open(file) as f:

      img = test_transform(f).unsqueeze(0)

      with torch.no_grad():

          out = model(img.to(device)).cpu().numpy()

          for key, value in ids.items():

              if value == np.argmax(out):

                    #name = classes[int(key)]

                    print(f"Predicted Label:and Key {key} and value {value}")

          plt.imshow(np.array(f))

          plt.show()