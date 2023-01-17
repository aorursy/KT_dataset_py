# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import sys

import cv2 as cv

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import PIL

from PIL import Image

import pydicom

import tensorflow as tf





#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

os.chdir('/kaggle/input/')

dpath='/kaggle/input/siim-isic-melanoma-classification/'
train_csv=pd.read_csv('./siim-isic-melanoma-classification/train.csv')

train_csv
# read tfrec

tfrd=os.path.join(dpath,'./tfrecords/')

tfr=tf.data.TFRecordDataset(tfrd)
type(tfrd+('./test06-687.tfrec'))
# Create a dictionary describing the features.

image_feature_description = {

    'height': tf.io.FixedLenFeature([], tf.int64),

    'width': tf.io.FixedLenFeature([], tf.int64),

    'depth': tf.io.FixedLenFeature([], tf.int64),

    'label': tf.io.FixedLenFeature([], tf.int64),

    'image_raw': tf.io.FixedLenFeature([], tf.string),

}
def _parse_image_function(example_proto):

  # Parse the input tf.Example proto using the dictionary above.

  return tf.io.parse_single_example(example_proto, image_feature_description)



parsed_image_dataset = tfr.map(_parse_image_function)

parsed_image_dataset
# dcm to jpg and info

dcm_path='/kaggle/input/siim-isic-melanoma-classification/train/'

outpath='/kaggle/working/dcm_jpg/'

images_path=os.listdir(dcm_path)
# read dcm

dcdir=os.path.join(dpath,'train/')

dc1=pydicom.dcmread(dcdir+'ISIC_0074311.dcm')
dc1
# Create list of first 20 jpg

tjd='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

ims=[]

limit=20

processed=0

for i in os.listdir(tjd):

    impath=os.path.join(tjd,i)

    if os.path.isfile(impath):

        j=Image.open(impath)

        ims.append(j)

    processed += 1

    if processed > limit:

        break

ims
# Plot first 20 jpg

fig=plt.figure(figsize=(15,10))

columns=5;rows=4

for i in range(1,columns*rows+1):

    img=ims[i]

    fig.add_subplot(rows,columns,i)

    plt.imshow(img)

plt.show