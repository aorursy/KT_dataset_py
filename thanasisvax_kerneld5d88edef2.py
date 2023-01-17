# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/vgg16-weights-tf"))

# Any results you write to the current directory are saved as output.
# Import all the necessary libraries for the project

import csv

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sn

import pandas as pd

import pickle

import tensorflow as tf

from sklearn.datasets import load_files       

from keras.utils import np_utils

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

from PIL import ImageFile   

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D

from keras.layers import Dropout, Flatten, Dense, Activation

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD, RMSprop, Adam

from keras.applications.vgg16 import VGG16
datagen = ImageDataGenerator(

    rescale=1. / 255,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    rotation_range = 10, # data augmentation

    vertical_flip = True,

    horizontal_flip=True,

    validation_split=0.2)
train_files = datagen.flow_from_directory('../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train',target_size = (240 , 240), shuffle=True, batch_size = 32, subset="training")

valid_files = datagen.flow_from_directory('../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train',target_size = (240 , 240), shuffle=True, batch_size = 32, subset="validation")

test_files = datagen.flow_from_directory('../input/stanford-car-dataset-by-classes-folder/car_data/car_data/test',target_size = (240 , 240), batch_size = 32)
base_model = VGG16(weights = '../input/vgg16-weights-tf/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' ,

                      include_top = False,

                     input_shape = (240 , 240 , 3))

base_model.summary()
for layer in base_model.layers[:-2]:

    layer.trainable=False

for layer in base_model.layers:

    print(layer, layer.trainable)
VGG16_model_aug = Sequential()

VGG16_model_aug.add(base_model)

VGG16_model_aug.add(GlobalAveragePooling2D())

VGG16_model_aug.add(Dense(196, activation='softmax'))

VGG16_model_aug.summary()
VGG16_model_aug.load_weights('../input/weights-initial/weights.best.aug2_lr01.hdf5')
for layer in VGG16_model_aug.layers[:-2]:

    layer.trainable=False

for layer in VGG16_model_aug.layers:

    print(layer, layer.trainable)
##VGG16_model_aug.load_weights('../input/weights-learn/weights.best.aug2_lr01.hdf5')
rmsprpo=RMSprop(lr=0.00001)

VGG16_model_aug.compile(loss='categorical_crossentropy', optimizer= rmsprpo, metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='weights.best.aug2_lr02.hdf5', 

                               verbose=1, save_best_only=True)
history3=VGG16_model_aug.fit_generator(train_files, 

          validation_data=valid_files,

       epochs=30, validation_steps=1546/32, steps_per_epoch =6598/32, callbacks=[checkpointer], verbose=1)
VGG16_model_aug.load_weights('weights.best.aug2_lr02.hdf5')
# summarize history for accuracy

plt.plot(history3.history['acc'])

plt.plot(history3.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history3.history['loss'])

plt.plot(history3.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
score = VGG16_model_aug.evaluate_generator(test_files, steps=len(test_files)/32,  workers=1, pickle_safe=True)

print('\n', 'Test accuracy:', score[1]*100)