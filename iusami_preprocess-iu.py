# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

IMG_SIZE = 512

NUM_CLASSES = 5

SEED = 77

TRAIN_NUM = 1000 # use 1000 when you just want to explore new idea, use -1 for full train
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
sns.countplot(train_df['diagnosis'])
test_df.head()
df0 = train_df[train_df['diagnosis']==0].sample(10,random_state=SEED)

df1 = train_df[train_df['diagnosis']==1].sample(10,random_state=SEED)

df2 = train_df[train_df['diagnosis']==2].sample(10,random_state=SEED)

df3 = train_df[train_df['diagnosis']==3].sample(10,random_state=SEED)

df4 = train_df[train_df['diagnosis']==4].sample(10,random_state=SEED)
#plt.imshow('../input/train_images/7b20210d9120.png')

skimage.io.imshow('../input/train_images/7b20210d9120.png')
fig = plt.figure(figsize=(20,20))

for count, i in enumerate(df0['id_code']):

    ax1 = fig.add_subplot(2,5,count+1)

    skimage.io.imshow('../input/train_images/' + str(i) + '.png')

plt.show()
fig = plt.figure(figsize=(20,20))

for count, i in enumerate(df1['id_code']):

    ax1 = fig.add_subplot(2,5,count+1)

    skimage.io.imshow('../input/train_images/' + str(i) + '.png')

plt.show()
fig = plt.figure(figsize=(20,20))

for count, i in enumerate(df2['id_code']):

    ax1 = fig.add_subplot(2,5,count+1)

    skimage.io.imshow('../input/train_images/' + str(i) + '.png')

plt.show()
fig = plt.figure(figsize=(20,20))

for count, i in enumerate(df3['id_code']):

    ax1 = fig.add_subplot(2,5,count+1)

    skimage.io.imshow('../input/train_images/' + str(i) + '.png')

plt.show()
fig = plt.figure(figsize=(20,20))

for count, i in enumerate(df4['id_code']):

    ax1 = fig.add_subplot(2,5,count+1)

    skimage.io.imshow('../input/train_images/' + str(i) + '.png')

plt.show()
fig = plt.figure(figsize=(30,15))

for count, i in enumerate(df4['id_code']):

    ax1 = fig.add_subplot(2,5,count+1)

    img = skimage.io.imread('../input/train_images/' + str(i) + '.png')

    gray = skimage.color.rgb2gray(img)

    sns.heatmap(gray,cbar=False,xticklabels=False, yticklabels=False)

    if count == 5:

        break

plt.show()
fig = plt.figure(figsize=(30,15))

for count, i in enumerate(df0['id_code']):

    ax1 = fig.add_subplot(2,5,count+1)

    img = skimage.io.imread('../input/train_images/' + str(i) + '.png')

    gray = skimage.color.rgb2gray(img)

    sns.heatmap(gray,cbar=False,xticklabels=False, yticklabels=False)

    if count == 5:

        break

plt.show()
def imghistplotter(imglist, normflag):

    fig = plt.figure(figsize=(30,20))

    for count,img in enumerate(imglist):

        ax = fig.add_subplot(2,len(imglist)/2,count+1)

        img_b = img[:,:,0].reshape(img.shape[0]*img.shape[1])

        img_g = img[:,:,1].reshape(img.shape[0]*img.shape[1])

        img_r = img[:,:,2].reshape(img.shape[0]*img.shape[1])

        plt.hist(img_r,bins=255,color='red',normed=normflag)

        plt.hist(img_g,bins=255,color='green',normed=normflag)

        plt.hist(img_b,bins=255,color='blue',normed=normflag)

    plt.show()
imglist = [cv2.imread('../input/train_images/' + str(i) + '.png') for i in df0['id_code']]
imghistplotter(imglist, normflag=True)
imglist_4 = [cv2.imread('../input/train_images/' + str(i) + '.png') for i in df4['id_code']]
imghistplotter(imglist_4,normflag=True)