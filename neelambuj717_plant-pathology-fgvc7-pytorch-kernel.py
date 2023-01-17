import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from sklearn.metrics import roc_auc_score

import os

import gc

from tqdm import tqdm

import matplotlib.pyplot as plt

import pydot

from keras.datasets import mnist

from tqdm import tqdm_notebook

import warnings

warnings.filterwarnings("ignore")

pd.options.display.max_columns = 999

from sklearn.model_selection import train_test_split

from scipy import stats



np.random.seed(42)

import tensorflow as tf

from numpy import random

import keras as k

from keras.layers import Dense, Flatten, Conv2D, Conv3D



import matplotlib.pylab as pylab

params = {'legend.fontsize': 'medium',

         'axes.labelsize': 'x-large',

         'axes.titlesize':'x-large',

         'xtick.labelsize':'x-large',

         'ytick.labelsize':'x-large'}



from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D, TimeDistributed, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from numpy import argmax, array_equal

import matplotlib.pyplot as plt

from keras.models import Model

# from imgaug import augmenters

from random import randint

pd.set_option('float_format', '{:.3f}'.format)

pylab.rcParams.update(params)

plt.rcParams['figure.figsize'] = (15, 6)

warnings.filterwarnings("ignore")

pd.options.display.max_columns = 999

pd.options.display.max_columns = 99



import torch 

import torchvision

from pathlib import Path
train_labels = pd.read_csv(r'../input/plant-pathology-2020-fgvc7/train.csv')

test_labels = pd.read_csv(r'../input/plant-pathology-2020-fgvc7/test.csv')

image_path = Path(r'../input/plant-pathology-2020-fgvc7/images')

train_images = train_labels['image_id'].tolist()

test_images = test_labels['image_id'].tolist()
import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

import torch.optim as optim

from torchvision.transforms import transforms

import torch.nn.functional as F
class PlantsData(Dataset):

    def __init__(self, directory, train_image_list,  size, labels, id_cols, target_cols, transform, img_aug):

        self.directory = directory

        self.labels = labels

        self.id_cols = id_cols

        self.target_cols = target_cols

        self.size = size

        self.train_image_list = train_image_list

        self.transform = transform

        self.aug = img_aug

        

    def __len__(self):

        return len(self.train_image_list)

    

    def __getitem__(self,idx):

        image_name = self.train_image_list[idx]

        image_path = os.path.join(self.directory, image_name+".jpg")

        img = cv2.imread(image_path)

        img = cv2.resize(img, (self.size, self.size))

        #Change background colurs for the image

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        if self.transform:

            img = self.aug(img)

#         img = img/255.0

        target = self.labels[self.labels[self.id_cols] == image_name][self.target_cols].values

        

        return img, target
in_channels = 3

learning_rate = 1e-3

batch_size = 32

num_epochs = 25

num_classes = 4
# Replacing the layers in the mdoels

class Identity(nn.Module):

    def __init__(self):

        super(Identity, self).__init()

        

    def forward(self, x):

        return x

    

    

# loading the model

model = torchvision.models.vgg16(pretrained=True)
class PlantNet(nn.Module):

    def __init__(self, n_channels, num_classes):

        super(PlantNet, self).__init__()

        self.n_channels = n_channels

        self.num_classes = num_classes

        self.VGG = torchvision.models.vgg16(pretrained=True)

        self.conv1 = nn.Sequential(nn.Conv2d(512, 1024,  kernel_size = 3, stride = 1, padding = 1), nn.BatchNorm2d(1024), nn.ReLU())

#         self.conv2 = nn.Sequential(nn.Conv2d(512, 1024,  kernel_size = 3, stride = 1, padding = 1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True))

        self.linear = nn.Sequential(nn.Linear(1024 * 8 * 8, 4096 ), nn.ReLU())

        self.final_layer = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_classes), nn.Sigmoid())

        

    def forward(self, x):

        feature = self.VGG.features

        vgg_features = feature(x)

        out = self.conv1(vgg_features)

#         out = self.conv2(out)

        out = out.reshape(out.shape[0], -1)

        out = self.linear(out)

        output = self.final_layer(out)

        return output
len(train_images)
my_trans = transforms.Compose([transforms.ToPILImage(),

                   transforms.RandomHorizontalFlip(p = 0.5), 

                   transforms.ColorJitter(brightness = 0.7),

                   transforms.RandomRotation(degrees = 45),

                   transforms.RandomVerticalFlip(p = 0.8),

                   transforms.ToTensor()])



train_loader = PlantsData(image_path, train_images[:1000], 256, train_labels, 'image_id', train_labels.columns[1:], transform = True, img_aug=my_trans)

train_plant_loader = DataLoader(train_loader, batch_size = 32, shuffle = True)



val_loader = PlantsData(image_path, train_images[1000: 1320], 256, train_labels, 'image_id', train_labels.columns[1:],  transform = True, img_aug=my_trans)

val_plant_loader = DataLoader(val_loader, batch_size = 32, shuffle = True)
from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm, tqdm_notebook
model = PlantNet(n_channels=3, num_classes = 4)

device = 'cuda'

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, verbose = True, factor = 0.2)
val_outputs = []

val_targets = []





def save_checkpoint(checkpoint, filename = 'plant_models.path.tar'):

    torch.save(checkpoint, filename)

filename = 'plant_models.path.tar'



mean_loss = np.nan

losses = []

f1_score_list = []

accuracy_list = []



for epoch in range(num_epochs):



    checkpoint = {'state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}

    if epoch > 1:

        if f1_score_list[epoch-1] > f1_score_list[epoch-2]:

            print(f"F1 score on validation set increased form {f1_score_list[epoch - 2]} to {f1_score_list[epoch-1]} saving the model as {filename}")

            save_checkpoint(checkpoint, filename = filename)



    loop = tqdm(enumerate(train_plant_loader), position = 0, total = len(train_plant_loader), leave = True)



    for i, (image, target) in loop:

#         image = image.permute(0 ,3, 1 ,2)

        image = image.to(device)

        target = target.to(device)

        

        #Feeding the images to the model

        output = model(image.float())

        loss = criterion(output, torch.argmax(target.squeeze(), axis = 1))

        

        # Propogating the loss backwards

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loop.set_description(f'Epoch {epoch+1}/{num_epochs}')

        loop.set_postfix(loss = loss.item(), mean_loss = mean_loss, accuracy_score = accuracy_score(val_targets, val_outputs) , F1_score = f1_score(val_targets, val_outputs, average = 'weighted'))

        losses.append(loss.item())

  

    model.eval()

    with torch.no_grad():

        val_outputs = []

        val_targets = []

        for i, (image, target) in enumerate(val_plant_loader):

#             image = image.permute(0,3,1,2)

            image = image.to(device)

            target = target.to(device)



            # Predictions from Model

            outputs = model(image.float())

            outputs = torch.argmax(outputs, axis = 1)

            outputs = outputs.to('cpu')

            val_outputs += outputs.numpy().tolist()



            target = torch.argmax(target.squeeze(), axis = 1)

            target = target.to('cpu')

            val_targets += target.numpy().tolist() 

            

    

    f1_score_list.append(f1_score(val_targets, val_outputs, average = 'weighted'))

    accuracy_list.append(accuracy_score(val_targets, val_outputs))

      

            

    mean_loss = sum(losses)/len(losses)

    scheduler.step(mean_loss)
criterion = nn.CrossEntropyLoss()

output = torch.randn(10, 120).float()

target = torch.FloatTensor(10).uniform_(0, 120).long()

loss = criterion(output, target)
output.shape, target.shape