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
!wget https://raw.githubusercontent.com/Iamsdt/60daysofudacity/master/day22/Helper.py
import Helper
os.listdir()
import torch

from torchvision import datasets, transforms,models

from torch.utils.data import DataLoader



data_dir = "../input/data/natural_images"



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



train_loader, test_loader = Helper.prepare_loader(data_dir, data_dir,

                                                  train_transform, test_transform, batch_size=200)



print(len(train_loader))

print(len(test_loader))
classes = os.listdir(data_dir)

len(classes)
Helper.visualize(test_loader, classes)
model = models.densenet161(pretrained=True)

model.classifier
model = Helper.freeze_parameters(model)
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2208, out_features=2208),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=2208, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.3),

  nn.Linear(in_features=1024, out_features=8),

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
from PIL import Image

from matplotlib import pyplot as plt

file = data_dir+'/car/car_0000.jpg'

print(file)



test(file)