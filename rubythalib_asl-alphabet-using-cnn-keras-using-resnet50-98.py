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
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Add, Input, ZeroPadding2D, AveragePooling2D
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt
import cv2
from glob import glob
from numpy import floor
import random
from numpy.random import seed
seed(1)
def plot_three_samples(letter):
    print("ASL Alphabet for letter: "+letter)
    base_path = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
    img_path = base_path + letter + '/**'
    path_contents = glob(img_path)
    
    plt.figure(figsize=(16,16))
    imgs = random.sample(path_contents,3)
    plt.subplot(1,3,1)
    plt.imshow(cv2.imread(imgs[0]))
    plt.subplot(1,3,2)
    plt.imshow(cv2.imread(imgs[1]))
    plt.subplot(1,3,3)
    plt.imshow(cv2.imread(imgs[2]))
    
    return

plot_three_samples('S')
path = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
path_test = '../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test'
target_size = (64,64)
target_dims = (64,64,3)
val_frac = 0.1
n_classes = 29
batch_size = 64

image_generator = ImageDataGenerator(samplewise_center = True, samplewise_std_normalization = True, validation_split=val_frac)

train_gen = image_generator.flow_from_directory(path, target_size=target_size, batch_size=batch_size, shuffle=True, subset='training')
val_gen = image_generator.flow_from_directory(path, target_size=target_size, subset='validation')
def identity_block(X,f,filters, stage, block):
  #defining name basis
  conv_name_base = 'res' +str(stage)+block+'_branch'
  bn_name_base = 'bn' +str(stage)+block+'_branch'

  #Retrive Filters
  F1,F2,F3 = filters

  X_shortcut = X

  X = Conv2D(filters=F1, kernel_size=(1,1), strides = (1,1), padding='valid', name = conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
  X = Activation('relu')(X)

  X = Conv2D(filters=F2, kernel_size=(f,f), strides = (1,1), padding='same', name = conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
  X = Activation('relu')(X)

  X = Conv2D(filters=F3, kernel_size=(1,1), strides = (1,1), padding='valid', name = conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)

  return X
def convolutional_block(X, f, filters, stage, block, s=2):
  conv_name_base = 'res' +str(stage)+block+'_branch'
  bn_name_base = 'bn' +str(stage)+block+'_branch'

  F1,F2,F3 = filters

  X_shortcut = X

  X = Conv2D(filters=F1, kernel_size=(1,1), strides = (s,s), name = conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
  X = Activation('relu')(X)

  X = Conv2D(filters=F2, kernel_size=(f,f), strides = (1,1),padding='same', name = conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
  X = Activation('relu')(X)

  X = Conv2D(filters=F3, kernel_size=(1,1), strides = (1,1),padding='valid', name = conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)

  X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides = (s,s), name = conv_name_base+'1',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
  X_shortcut = BatchNormalization(axis = 3, name = bn_name_base+'1')(X_shortcut)
  X = Add() ([X, X_shortcut])
  X = Activation('relu')(X)

  return X
def ResNet50(input_shape = (64,64,3), classes = 29):
  X_input = Input(input_shape)

  #Zero padding
  X = ZeroPadding2D((3,3))(X_input)

  #stage 1
  X = Conv2D(64,(7,7),strides=(2,2), name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name = 'bn_conv1')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((3,3), strides=(2,2))(X)

  #stage 2
  X = convolutional_block(X, f=3, filters=[64, 64, 256], stage = 2, block='a', s=1)
  X = identity_block(X, 3, [64,64,256], stage=2, block='b')
  X = identity_block(X,3,[64,64,256], stage = 2, block = 'c')

  #stage 3
  X = convolutional_block(X, f=3, filters=[128, 128, 512], stage = 3, block='a', s=2)
  X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='b')
  X = identity_block(X,3,filters=[128, 128, 512], stage = 3, block = 'c')
  X = identity_block(X,3,filters=[128, 128, 512], stage = 3, block = 'd')

  #stage 4
  X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage = 4, block='a', s=2)
  X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='b')
  X = identity_block(X,3,filters=[256, 256, 1024], stage = 4, block = 'c')
  X = identity_block(X,3,filters=[256, 256, 1024], stage = 4, block = 'd')
  X = identity_block(X,3,filters=[256, 256, 1024], stage = 4, block = 'f')

  #stage 5
  X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage = 5, block='a', s=2)
  X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block='b')
  X = identity_block(X,3,filters=[512, 512, 2048], stage = 5, block = 'c')

  X = AveragePooling2D((2,2), name = 'avg_pool')(X)

  X = Flatten()(X)
  X = Dense(classes, activation='softmax', name='fc'+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

  model = Model(inputs = X_input, outputs = X, name='ResNet50')

  return model
model = ResNet50()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
history = model.fit_generator(train_gen,epochs=5, validation_data=val_gen)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.show()