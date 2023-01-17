# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np

import glob

from sklearn import cluster

from scipy.misc import imread

import cv2

import skimage.measure as sm

# import progressbar

import multiprocessing

import random

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

new_style = {'grid': False}

plt.rc('axes', **new_style)



# Function to show 4 images

def show_four(imgs, title):

    #select_imgs = [np.random.choice(imgs) for _ in range(4)]

    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(4)]

    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 3))

    plt.suptitle(title, size=20)

    for i, img in enumerate(select_imgs):

        ax[i].imshow(img)



# Function to show 8 images

def show_eight(imgs, title):

    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(8)]

    _, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(20, 6))

    plt.suptitle(title, size=20)

    for i, img in enumerate(select_imgs):

        ax[i // 4, i % 4].imshow(img)
from PIL import Image

sizes=[]

train_files = sorted(glob.glob('../input/train/*/*.jpg'))

for img in train_files:

    with Image.open(img) as im:

        width, height = im.size

        sizes.append((str)(width)+','+(str)(height))

train = np.array(sizes)

train

#print('Length of train {}'.format(len(train)))
print('Sizes in train:')



pd.Series(train).value_counts()
import xgboost