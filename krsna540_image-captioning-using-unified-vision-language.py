# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf



# You'll generate plots of attention in order to see which parts of an image

# our model focuses on during captioning

import matplotlib.pyplot as plt



# Scikit-learn includes many helpful utilities

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



import collections

import random

import re

import numpy as np

import os

import time

import json

from glob import glob

from PIL import Image

import pickle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
header_list = ["image_name", "comment_number", "comment"]

annotations_df=pd.read_csv("../input/flickr-image-dataset/flickr30k_images/results.csv",delimiter='|',names=header_list)
annotations_df.drop(annotations_df.head(3).index, inplace=True)

annotations_df.head()
from IPython.display import display, Image

display(Image(filename='/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/flickr30k_images/1000092795.jpg'))

print(annotations_df.loc[0][2])
annotations_df.sample(frac=1)

annotations_df= annotations_df.head(5000)
import cv2 as cv

import cv2 



IMAGE_SIZE = (299, 299)

dataset_path = '/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/flickr30k_images/'

train_images = []

train_captions = []

for index, row in tqdm(annotations_df.iterrows()): 

    train_captions.append(row['comment'])

    img_path=dataset_path+row['image_name']

    img = tf.io.read_file(img_path)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (299, 299))

    img = tf.keras.applications.inception_v3.preprocess_input(img)

    train_images.append(image)

    