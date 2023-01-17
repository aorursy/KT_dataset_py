# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in

 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from random import randint

 

import scipy.ndimage as ndimage

 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as keras

import skimage.io as io

import skimage.transform as trans

from PIL import Image

import os

import sys

import random

import math

import re

import time

import cv2

import matplotlib

import matplotlib.pyplot as plt



import keras

import keras.backend as K

import keras.layers as KL

import keras.engine as KE

import keras.models as KM



def identity_block(input_tensor, kernel_size, filters, use_bias = True, train_bn = True):

    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1), use_bias = use_bias)(input_tensor)

    x = BatchNormalization()(x, training = train_bn)

    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding = 'same', use_bias = use_bias)(x)

    x = BatchNormalization()(x, training = train_bn)

    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), use_bias = use_bias)(x)

    x = BatchNormalization()(x, training = train_bn)

    x = KL.Add()([x, input_tensor])

    x = Activation('relu')(x)

    return x



def conv2d_block(input_tensor, n_filters, filters, strides = (1,1), kernel_size = 3, batchnorm = True, train_bn = True):    

    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding = 'same')(x)

    if batchnorm:

        x = BatchNormalization()(x)

    shortcut = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)

    shortcut = BatchNormalization()(shortcut)

    x = KL.Add()([x,shortcut])

    x = Activation('relu')(x)

    return x

   

def resnet_graph(input_image, train_bn = True, stage = 2):

    x = KL.ZeroPadding2D((17, 17))(input_image)

    x = KL.Conv2D(64, (16, 16), strides = (2, 2), use_bias = True)(x)

    x = BatchNormalization(name='bn_conv1')(x, training = train_bn)

    x = KL.Activation('relu')(x)

    C1 = x = KL.MaxPooling2D((31, 31), strides = (2, 2), padding = 'same')(x)

    x = conv2d_block(x, 16, [64, 64, 16], strides = (1, 1), train_bn = train_bn)

    x = identity_block(x, 3, [64, 64, 16], train_bn = train_bn)

    C2 = x = identity_block(x, 3, [64, 64, 16], train_bn = train_bn)

    x = conv2d_block(x, 16, [64, 64, 16], train_bn = train_bn)

    x = identity_block(x, 3, [64, 64, 16], train_bn = train_bn)

    x = identity_block(x, 3, [64, 64, 16], train_bn = train_bn)

    C3 = x = identity_block(x, 3, [64, 64, 16], train_bn = train_bn)

    x = conv2d_block(x, 16, [64, 64, 16], train_bn = train_bn)

    for i in range(50):

        x = identity_block(x, 3, [64, 64, 16], train_bn = train_bn)

    C4 = x

    outputs = Conv2D(1, (1, 1), activation = 'sigmoid') (C4)

    model = Model(inputs = [input_img], outputs = [outputs])

    return model

    

def make_pairs(nums):

    i = 0

    pairs = list()

    while((i + 1) < len(nums)):

        pairs.append((nums[i], nums[i + 1]))

        i += 2

    return pairs

   

def str_to_mask(size, str):

    toks = str.split(' ')

    nums = list(map(int, toks))

    mask = np.zeros(size)

    pix_pairs = make_pairs(nums)

    for p in pix_pairs:

        i = p[0] - 1

        while(i <= p[0] + p[1] - 1 ):

            i_r = i % size[0]

            i_c = int(i / size[0])

            if(i_r >= size[0] or i_c >= size[1]):

                i += 1

                continue

            mask[i_r, i_c] = 1

            i += 1

    return mask

 

def mygenerator(table):

    while 1:

        for i, x in table.iterrows():

            path = '../input/train_images/train_images/' + x[0] + '/images/' + x[0] + '.png'

            X = io.imread(path, as_gray = True)

            y = str_to_mask(X.shape, x[1])

            X = trans.resize(X, (16, 16), anti_aliasing = True)

            y = trans.resize(y, (16, 16), anti_aliasing = True)

            rot = randint(0, 3) * 90

            X = trans.rotate(X, rot)

            y = trans.rotate(y, rot)

            X = np.expand_dims(X, axis = 2)

            X = np.expand_dims(X, axis = 0)

            y = np.expand_dims(y, axis = 2)

            y = np.expand_dims(y, axis = 0)

            yield (X, y)

 

def mygenerator_batch(table, bs):

    gen = mygenerator(table)

    while 1:

        train = []

        mask = []

        for i in range(bs):

            X, y = next(gen)

            train.append(X)

            mask.append(y)

        yield (np.concatenate(train, axis = 0), np.concatenate(mask, axis = 0))



def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b > prev + 1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths

 

 

def clean_mask(imgi, tol):

    img = np.copy(imgi)

    rl = img.shape[0]

    cl = img.shape[1]

    for r in range(rl):

        c = 0

        while c < cl:

            if(img[r,c] >= tol):

                img[r,c] = 1

            else:

                img[r,c] = 0

            c += 1

    return img



data_table = pd.read_csv('../input/train_masks.csv')

tb_split = int(data_table.shape[0] * 0.60)

 

train_table = data_table[0:tb_split]

val_table = data_table[tb_split:]



input_img = Input((9, 9, 1), name = 'img')

model = resnet_graph(input_img)

model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])

stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss')

history = model.fit_generator(mygenerator_batch(train_table, 8), callbacks=[stop_callback], steps_per_epoch=1, epochs=100, validation_data=mygenerator_batch(val_table, 8), validation_steps=1, verbose=2, shuffle=True)



sample = mygenerator(train_table)

sX, sY = next(sample)

spX = sX[0,:,:,0]

spY = sY[0,:,:,0]

res = model.predict(sX)

res = res[0,:,:,0]



test_files = os.listdir('../input/test_images/test_images')



def a_to_str(a):

    s = ''

    for x in a:

        s += str(x) + ' '

    return s

 

res = []

for x in test_files:

    path = '../input/test_images/test_images/' + x + '/images/' + x + '.png'

    img = io.imread(path, as_gray = True)

    img = trans.resize(img, (64,64), anti_aliasing = True)

    original_shape = img.shape

    img = np.expand_dims(img, axis = 2)

    img = np.expand_dims(img, axis = 0)

    mask = model.predict(img)

    mask = mask[0,:,:,0]

    mask = trans.resize(mask, original_shape, anti_aliasing = True)

    cmask = clean_mask(mask, 0.25)

    cmask = cmask[:, 0:-2]

    lables, nlables = ndimage.label(cmask)

    for ln in range(1, nlables+1):

        lables_mask = np.where(lables == ln, 1, 0)

        rle = rle_encoding(lables_mask)

        res.append([x, str(rle)])

 

out = pd.DataFrame(res, columns = ['ImageId','EncodedPixels'])

print(out)

out.to_csv('plswrk.csv', index = None)