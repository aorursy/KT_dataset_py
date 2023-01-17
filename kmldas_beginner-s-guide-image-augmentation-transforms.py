import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import cv2
import random
from random import randint
import time


import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from scipy import ndimage

import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder

from tqdm.notebook import tqdm

from sklearn.metrics import f1_score

DATA_DIR = '../input/dog-breed-identification'


TRAIN_DIR = DATA_DIR + '/train'                           
TEST_DIR = DATA_DIR + '/test'                             

TRAIN_CSV = DATA_DIR + '/labels.csv'                     
TEST_CSV = DATA_DIR + '/submission.csv' 
data_df = pd.read_csv(TRAIN_CSV)
data_df.head(10)
labels_names=data_df["breed"].unique()
labels_sorted=labels_names.sort()

labels = dict(zip(range(len(labels_names)),labels_names))
labels 

lbl=[]
for i in range(len(data_df["breed"])):
    temp=list(labels.values()).index(data_df.breed[i])
    lbl.append(temp)

    
data_df['lbl'] = lbl
#data_df['lbl'] = data_df['lbl'].astype(str)
data_df.head()
path_img=[]
for i in range(len(data_df["id"])):
    temp=TRAIN_DIR + "/" + str(data_df.id[i]) + ".jpg"
    path_img.append(temp)

data_df['path_img'] =path_img
data_df.head()
num_images = len(data_df["id"])
print('Number of images in Training file:', num_images)
no_labels=len(labels_names)
print('Number of dog breeds in Training file:', no_labels)
bar = data_df["breed"].value_counts(ascending=True).plot.barh(figsize = (30,120))
plt.title("Distribution of the Dog Breeds", fontsize = 20)
bar.tick_params(labelsize=16)
plt.show()
data_df["breed"].value_counts(ascending=False)
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data_df.path_img[i]))
    ax.set_title(data_df.breed[i])
plt.tight_layout()
plt.show()
random_img=randint(0,len(data_df.path_img))
img_path=data_df.path_img[random_img]
img= plt.imread(img_path)

plt.imshow(img)
plt.title("Original image")
plt.show()

plt.imshow(cv2.resize(img, (150,150)))
plt.title("After resizing")
plt.show()
random_img=randint(0,len(data_df.path_img))
img_path=data_df.path_img[random_img]
img= plt.imread(img_path)

plt.imshow(img)
plt.title("Original image")
plt.show()


#rotation angle in degree

rotated1 = ndimage.rotate(img, 90)
plt.imshow(rotated1)
plt.title("Image rotated 90 degrees")
plt.show()
random_img=randint(0,len(data_df.path_img))
img_path=data_df.path_img[random_img]
img= plt.imread(img_path)

plt.imshow(img)
plt.title("Original image")
plt.show()


img=cv2.resize(img, (150,150))
turn =90

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(ndimage.rotate(img, i*90))
    ax.set_title("After resizing rotated "+ str(i*90) +" degrees")
plt.tight_layout()
plt.show()
#imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
#this will resize the image 
    T.Resize(256),   
   
#Randomly change the brightness, contrast and saturation of an image
#    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),    

#this will remove parts (crop) the Image at a random location.   
#    T.RandomCrop(32, padding=4, padding_mode='reflect'),   

#Horizontally flip (rotate by 180 degree) the given image randomly; default is 50% of images
    T.RandomHorizontalFlip(), 
    
#Rotate the image by angle -here by 10%
    T.RandomRotation(10),
    
#convert it to a tensor   
    T.ToTensor()

#Normalize a tensor image with mean and standard deviation - here with the Imagenet stats
#    T.Normalize(*imagenet_stats,inplace=True), 
    
#Randomly selects a rectangle region in an image and erases its pixels.    
#    T.RandomErasing(inplace=True)
])

class DogDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['id'], row['lbl']
        img_fname = self.root_dir + "/" + str(img_id) + ".jpg"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, img_label
data_ds = DogDataset(data_df, TRAIN_DIR, transform=train_tfms)
def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', labels[target])
show_sample(*data_ds[241])
show_sample(*data_ds[149])