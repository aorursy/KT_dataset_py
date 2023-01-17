# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import tensorflow 



import keras

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense,Flatten

from keras.optimizers import Adam

from keras.optimizers import RMSprop

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import *

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.preprocessing import image

from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint

from keras.models import load_model



import matplotlib.pyplot as plt

import itertools

from matplotlib import pyplot as plt

%matplotlib inline

import glob

import matplotlib.image as mpimg

from keras.preprocessing import image

import imageio as im

import glob

import matplotlib

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import imageio as im

train_path = '/kaggle/input/plantvillageaug/aug_data/Aug_Data'
train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(128,128),classes =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10',

                                                                                                 'c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20',

                                                                                                    'c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30',

                                                                                                 'c_31','c_32','c_33','c_34','c_35','c_36','c_37'],batch_size=10)
test_path = '/kaggle/input/plantvillagepredictions/val/val'
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(128,128),classes =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10',

                                                                                                 'c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20',

                                                                                                    'c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30',

                                                                                                 'c_31','c_32','c_33','c_34','c_35','c_36','c_37'],batch_size=10)
def plots(ims, figsize=(12,6), rows=5, interp=False, titles=None):

    if type(ims[0]) is np.ndarray:

        ims = np.array(ims).astype(np.uint8)

        if (ims.shape[-1] != 3): ims = ims.transpose((0,2,3,1))

    f = plt.figure(figsize=figsize)

    for i in range(len(ims)):

        sp = f.add_subplot(rows, len(ims)//rows, i+1)

        sp.axis('Off')

        if titles is not None: sp.set_title(titles[i], fontsize=16)

        plt.imshow(ims[i], interpolation=None if interp else 'none')
img,labels = next(train_batches)
plots(img,titles=labels)
model = Sequential()



model.add(Conv2D(64,(3,3),input_shape=(128,128,3),padding='same'))

model.add(Activation('relu'))



model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu')) 



model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu')) 



model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu')) 



model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Conv2D(64,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(BatchNormalization())



model.add(Dense(128))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Dense(128))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Dense(38,activation='softmax'))

model.build()
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_batches,steps_per_epoch=8761,epochs=10)
model.save('Ultimate_Model.h5')
output = model.predict_generator(test_batches,steps=440)
classes1 = test_batches.classes
count = 0

for i in range(0,4396):

    for j in range(0,38):

        if output[i][j] == max(output[i]):

            if j == classes1[i]:

                count+=1

print(count)
count/4396
pwd