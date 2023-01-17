# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

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
def read_img(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

    return img
img1=read_img('../input/images/img1.jpeg')

img2=read_img('../input/images/img2.jpeg')

img3=read_img('../input/images/img3.jpeg')

img4=read_img('../input/images/img4.jpeg')
plt.figure(figsize=(20,5))

plt.subplot(1,4,1)

plt.imshow(img1)

plt.subplot(1,4,2)

plt.imshow(img2)

plt.subplot(1,4,3)

plt.imshow(img3)

plt.subplot(1,4,4)

plt.imshow(img4)
img_list=[img1,img2,img3,img4]
plt.figure(figsize=(20,5))

for i in range(1,5):

    image = cv2.cvtColor(img_list[i-1], cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image, 4 , cv2.GaussianBlur( image , (0 ,0 ) , 30) ,-4 ,128)

    plt.subplot(1,4,i)

    plt.imshow(image,cmap='gray')
plt.figure(figsize=(20,5))

for i in range(1,5):

    image = cv2.cvtColor(img_list[i-1], cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image, 4 , cv2.GaussianBlur( image , (0 ,0 ) , 30) ,-4 ,128)

    plt.subplot(1,4,i)

    plt.imshow(image)
# Unsharp Masking
plt.figure(figsize=(20,10))

for i in range(1,5):

    image = cv2.cvtColor(img_list[i-1], cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.GaussianBlur( image , (0 ,0 ) , 5) 

    plt.subplot(1,4,i)

    plt.imshow(image,cmap='gray')
plt.figure(figsize=(20,5))

for i in range(1,5):

    image = cv2.cvtColor(img_list[i-1], cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image, 4 , cv2.GaussianBlur( image , (0 ,0 ) , 30) ,-4 ,128)

    image = cv2.cvtColor(img_list[i-1], cv2.COLOR_BGR2HSV)

    print(image.shape)

    plt.subplot(1,4,i)

    plt.imshow(image)