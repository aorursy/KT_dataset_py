# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

from matplotlib import image

from matplotlib import pyplot as py

import numpy as np

from numpy import array

from tqdm import tqdm_notebook as tqdm

from matplotlib.pyplot import *



dir = '../input/tgs-salt-identification-challenge/train'


#import images into list(data_x)

images_dir = '{}/{}'.format(dir, '/images')

images = os.listdir(images_dir)

data_x = []

for image in tqdm(images):

    image_dir = '{}/{}'.format(images_dir, image)

    x = py.imread(image_dir)

    data_x.append(x)
#import masks into list(data_y)

masks_dir = '{}/{}'.format(dir, '/masks')

masks = os.listdir(masks_dir)

data_y = []

for mask in tqdm(masks):

    mask_dir = '{}/{}'.format(masks_dir, mask)

    y = py.imread(mask_dir)

    data_y.append(np.atleast_3d(y))
#train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.20, random_state = 42)

print(len(x_train))
def res(data):



    #rescale

    from skimage.transform import rescale

    # xem map(), lambda, tqdm

    data_rescaled = list(tqdm(map(lambda image: rescale(image, (64/101), anti_aliasing = True), data)))

    

    return data_rescaled

print(r"x_train element's shape is: {}, rescaling x_train...".format(x_train[1].shape))

py.subplot(1, 2, 1)

py.imshow(x_train[1])



#rescale x_train

x_train = res(x_train) 





#show x_train after rescale

print(r"new x_train element's shape is:", x_train[1].shape)

py.subplot(1, 2, 2)

py.imshow(x_train[1])



#rescale x_test

print('rescaling x_test...')

x_test = res(x_test)



#rescale y_train

print('rescaling y_train...')

y_train = res(y_train)



#resclae y_test

print('rescaling y_test...')

y_test = res(y_test)



print('rescaling finished!')
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.python.keras.models import Sequential, Model

from tensorflow.python.keras.layers.core import Dense, Activation

from tensorflow.python.keras.layers.convolutional import Convolution2D

from tensorflow.python.keras.layers import MaxPooling2D, Input, UpSampling2D, concatenate, Conv2DTranspose

from tensorflow.python.keras.losses import binary_crossentropy

from tensorflow.python.keras.optimizers import SGD

from tensorflow.python.keras import metrics

import numpy as np

import tensorflow as tf

import os

import skimage.io as io

import skimage.transform as trans

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as keras

def UNet():

    #input RBG (64, 64, 3)

    inp = Input((64, 64, 3))



    convolution1 = Convolution2D(16, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid')(inp)

    convolution1 = Convolution2D(16, (3, 3), padding = 'same', activation = 'sigmoid')(convolution1)#size = (64, 64, 16)

    maxpooling1 = MaxPooling2D(pool_size = (2, 2))(convolution1) #size = (32, 32, 16)



    conv2 = Convolution2D(32, (3, 3), padding = 'same', activation = 'sigmoid')(maxpooling1) 

    conv2 = Convolution2D(32, (3, 3), padding = 'same', activation = 'sigmoid')(conv2) #size = (32, 32, 32)

    drop2 = Dropout(0.4)(conv2)

    pool2 = MaxPooling2D(pool_size = (2, 2))(drop2) 

    #size = (16, 16, 32)



    conv3 = Convolution2D(64, (3, 3), padding = 'same', activation = 'sigmoid')(pool2) 

    conv3 = Convolution2D(64, (3, 3), padding = 'same', activation = 'sigmoid')(conv3) #size = (16, 16, 64)

    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3) 

    #size = (8, 8, 64)



    conv4 = Convolution2D(128, (3, 3), padding = 'same', activation = 'sigmoid')(pool3) #size = (8, 8, 128)

    drop4 = Dropout(0.4)(conv4)

    up3 = Conv2DTranspose(64, (3, 3), padding = 'same', strides = (2, 2))(drop4)

    #size = (16, 16, 64)



    skipconnection3 = concatenate([up3, conv3], axis = 3) #size = (16, 16, 128)

    conv3 = Convolution2D(64, (3, 3), padding = 'same', activation = 'sigmoid')(skipconnection3) 

    conv3 = Convolution2D(64, (3, 3), padding = 'same', activation = 'sigmoid')(conv3) #size = (16, 16, 64)

    up2 = Conv2DTranspose(32, (3, 3), padding = 'same', strides = (2, 2))(conv3)

    #size = (32, 32, 32)



    skcn2 = concatenate([up2, conv2], axis = 3) #size = (32, 32, 64)

    conv2 = Convolution2D(32, (3, 3), padding = 'same', activation = 'sigmoid')(skcn2) 

    conv2 = Convolution2D(32, (3, 3), padding = 'same', activation = 'sigmoid')(conv2) #size = (32, 32, 32)

    up1 = Conv2DTranspose(16, (3, 3), padding = 'same', strides = (2, 2))(conv2)

    #size = (64, 64, 16)





    skcn1 = concatenate([up1, convolution1], axis = 3) #size = (64, 64, 32)

    conv1 = Convolution2D(16, (3, 3), padding = 'same', activation = 'sigmoid')(skcn1) 

    conv1 = Convolution2D(16, (3, 3), padding = 'same', activation = 'sigmoid')(conv1) 

    #size = (64, 64, 16)



    conv0 = Convolution2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(conv1) 

    #size = (64, 64, 1)



    model = Model(inputs = inp, outputs = conv0)



    return model
loss = binary_crossentropy

optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

metrics = ['accuracy']

epochs = 15

batch_size = int(len(x_train)/10)

validation_data = (array(x_test), array(y_test))

print(array(x_test).shape)

print(array(y_test).shape)
model = UNet()

model.compile(

    loss = loss,

    optimizer = optimizer,

    metrics = metrics)

model.summary()
class myCallBack(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('loss') < 0.5):

            print('\n Loss is low so cancelling training')

            self.model.stop_training = True
callbacks = myCallBack()

model.fit(

    array(x_train), 

    array(y_train),

    batch_size,

    epochs,

    shuffle = True,

    validation_data = validation_data,

    callbacks=[callbacks])