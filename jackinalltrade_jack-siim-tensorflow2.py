# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd

import glob

import pydicom



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/siim-acr-pneumothorax-segmentation"))

print(os.listdir("../input/siim-test-train/siim"))

import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

from mask_functions import rle2mask



# Any results you write to the current directory are saved as output.
im_height = 1024

im_width = 1024

im_chan = 1



print('reading input images and mask dataset.....') 

df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample images/train-rle-sample.csv', header=None, index_col=0)



X_train = np.zeros((10, im_height, im_width, im_chan), dtype=np.uint8)

Y_train = np.zeros((10, im_height, im_width, 1), dtype=np.bool)



for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')):

    dataset = pydicom.dcmread(file_path)

    X_train[q] = np.expand_dims(dataset.pixel_array, axis=2)

    if df.loc[file_path.split('/')[-1][:-4],1] != '-1':

        Y_train[q] = np.expand_dims(rle2mask(df.loc[file_path.split('/')[-1][:-4],1], 1024, 1024).T, axis=2)

    else:

        Y_train[q] = np.zeros((1024, 1024, 1))

print('done!')      
from skimage.color import label2rgb

fig = plt.figure(figsize=(30, 30))

print('Preparing for visualization...')

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')):

    plt.subplot(3,10,q+1)

    plt.title('Input Images')

    plt.imshow(X_train[q,:,:,0],cmap='gray')

    plt.axis('off')

    

    plt.subplot(3,10,q+11)    

    plt.title('mask Images')

    plt.imshow(Y_train[q,:,:,0])

    plt.axis('off')

    

    plt.subplot(3,10,q+21)

    plt.title('overlap Images')

    plt.imshow(label2rgb(Y_train[q,:,:,0], image = X_train[q,:,:,0],alpha = 0.3))

    plt.axis('off') 

print('done')    
#!pip install tensorflow-gpu==2.0.0-alpha0

import tensorflow as tf

tf.__version__
import random

import warnings



from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Dropout, Lambda

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K



import tensorflow as tf





warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

#np.random.seed = seed
# Build U-Net model

inputs = Input((im_height, im_width, im_chan))

s = Lambda(lambda x: x / 255) (inputs)



c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)

c1 = BatchNormalization()(c1)

c1 = Dropout(0.1) (c1)

c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)

c1 = BatchNormalization()(c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)

c2 = BatchNormalization()(c2)

c2 = Dropout(0.1) (c2)

c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)

c2 = BatchNormalization()(c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)

c3 = BatchNormalization()(c3)

c3 = Dropout(0.2) (c3)

c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)

c3 = BatchNormalization()(c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)

c4 = BatchNormalization()(c4)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)

c4 = BatchNormalization()(c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)

c5 = BatchNormalization()(c5)

c5 = Dropout(0.3) (c5)

c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

c5 = BatchNormalization()(c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)

c6 = BatchNormalization()(c6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

c6 = BatchNormalization()(c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)

c7 = BatchNormalization()(c7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

c7 = BatchNormalization()(c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)

c8 = BatchNormalization()(c8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

c8 = BatchNormalization()(c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)

c9 = BatchNormalization()(c9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

c9 = BatchNormalization()(c9)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
tf.keras.utils.plot_model(model,to_file='unet_model.png',show_shapes=True,show_layer_names=True,rankdir='TB')