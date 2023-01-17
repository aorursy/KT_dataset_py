#Data Manipulation Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re #regular expressions

#Progress bar

from tqdm import tqdm

from datetime import datetime

#Read Images

import os

from skimage import io

#from skimage import io #returning error ImportError: cannot import name 'io' so temporarily commented

from PIL import Image

import cv2 # When open cv was used, there was an error in getting array from image. Using Pillow eliminated the error.



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns



#Image copy

from shutil import copyfile

from random import seed

from random import random

import shutil





#Model Pre-processing

#from sklearn.model_selection import train_test_split



#Modelling

import tensorflow as tf

import sys

from matplotlib import pyplot

from keras.models import Sequential

from keras.utils import to_categorical

from keras.applications.vgg16 import VGG16

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.models import Model

from keras.optimizers import SGD, Adam

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import  r2_score,roc_auc_score,f1_score,recall_score,precision_score,classification_report, confusion_matrix,log_loss

import random





# Image load

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import load_model
image_path = '../input/simi2020/UmojaHack#1_SAEON_Identifying_marine_invertebrates/UmojaHack#1_SAEON_Identifying_marine_invertebrates/train_small/train_small'

train_directories = os.listdir(image_path)

#Query Image List

image_categories = []

file_names =[]

image_names = []

# Loop across the directories having images.

for category in train_directories:        

    #full_image_path = image_path +  category + "/" +category + "/"

    full_image_path = image_path + "/" +category + "/"

    image_file_names = [os.path.join(full_image_path, f) for f in os.listdir(full_image_path)] # Retrieve the filenames from the all the  directories. OS package used.

    for file in image_file_names:         # Read the labels and load them into an array

        file_name = os.path.basename(file) ## Eliminate path from file name

        image_categories.append(category)

        file_names.append(file)

        image_names.append(file_name)

        

df = pd.DataFrame({'file_names': file_names, 'image_names': image_names,'image_categories':image_categories}, columns=['file_names', 'image_names','image_categories'])

        

#Delete directory if it exists.

def ignore_absent_file(func, path, exc_inf):

    except_instance = exc_inf[1]

    if isinstance(except_instance, FileNotFoundError):

        return

    raise except_instance



shutil.rmtree('/kaggle/working/SAEON2', onerror=ignore_absent_file)





# create directories

dataset_home = 'SAEON2/'

subdirs = ['train_aug/', 'validation/']

for subdir in subdirs:

    # create label subdirectories

    #labeldirs = ['train_elephants/', 'train_zebras/']

    for labldir in train_directories:

        newdir = dataset_home + subdir + labldir

        os.makedirs(newdir, exist_ok=True)

        

# Copy files from input to output train and validaton directories and their corresponding class directories

import random

seed = 1

val_ratio = 0.25

for index, row in df.iterrows():

    if row['image_categories'] != 'test':

        src = row['file_names']

        if random.random() < val_ratio:

            dst = '/kaggle/working/SAEON2/validation'+ '/' + row['image_categories'] + '/' +row['image_names']

        else:

            dst = '/kaggle/working/SAEON2/train'+ '/' + row['image_categories'] + '/' +row['image_names']

    copyfile(src, dst)
dataset_home = 'SAEON2/'

subdirs = ['train_aug/']

for subdir in subdirs:

    # create label subdirectories

    #labeldirs = ['train_elephants/', 'train_zebras/']

    for labldir in train_directories:

        newdir = dataset_home + subdir + labldir

        os.makedirs(newdir, exist_ok=True)

df[0:1]
import imgaug.augmenters as iaa

seq = iaa.Sequential([

    iaa.Affine(rotate=(-25, 25)),

    iaa.AdditiveGaussianNoise(scale=(10, 60)),

    iaa.Crop(percent=(0, 0.2)),

    iaa.AddElementwise((-40, 40), per_channel=0.5),

    #iaa.AdditiveLaplaceNoise(scale=0.2*255,per_channel=True),

    #iaa.AdditivePoissonNoise(scale=40, per_channel=True),

    iaa.Multiply((0.5, 1.5), per_channel=0.5),

    #iaa.Cutout(fill_mode="gaussian", fill_per_channel=True),

    #iaa.Dropout2d(p=0.5, nb_keep_channels=1),

    iaa.SaltAndPepper(0.1, per_channel=True),

    iaa.Invert(0.25, per_channel=0.5)

    ])



for index, row in df[0:1].iterrows():

    images_aug = seq(images=row['file_names'])



print("Augmented:")

ia.imshow(np.hstack(images_aug))
