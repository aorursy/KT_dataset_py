# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cats_and_dogs-20190503t090706z-001/Cats_and_Dogs"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from keras.applications import VGG16

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
# Initialize the folders with train, test and validation datasets (in "/My Drive/..." or from your local repository where you have downloaded data):



train = '../input/cats_and_dogs-20190503t090706z-001/Cats_and_Dogs/train'

val =   '../input/cats_and_dogs-20190503t090706z-001/Cats_and_Dogs/val'

test =  '../input/cats_and_dogs-20190503t090706z-001/Cats_and_Dogs/test'



# The shape of the RGB image

img_width, img_height, channels = 150, 150, 3 # you can try different sizes



# input shape

input_shape = (img_width, img_height, 3)

# position matters!

# Number_of_channels can be at the first or the last position

# in our case - "channels last"



# batch size

batch_size = 64

# train set size

nb_train_samples = 20000

# validation set size 

nb_validation_samples = 2500

# test set size

nb_test_samples = 2500
datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = datagen.flow_from_directory(

    train,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')



val_generator = datagen.flow_from_directory(

    val,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')



test_generator = datagen.flow_from_directory(

    test,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
model = Sequential()



# 1: +Convolutional

# For example:

model.add(Conv2D(16, (3, 3), input_shape=(150, 150, 3)))

model.add(Activation('relu'))



# 2: +Pooling

model.add(MaxPooling2D(pool_size=(2,2)))

# 3:

model.add(Conv2D(32, (3, 3)))

#     +Relu

model.add(Activation('relu'))

# 4:  +Pooling 

model.add(MaxPooling2D(pool_size = (2,2)))

# 5:  +Convolutional

model.add(Conv2D(64, (3,3)))

#     +Relu

model.add(Activation('relu'))

# 6:  +Pooling

model.add(MaxPooling2D(pool_size = (2,2)))

# 7:  +Flattening

model.add(Flatten())

# 8:  +Convolutional

model.add(Dense(64, activation = 'relu'))

#     +ReLu

# 9:  +Dropout

model.add(Dropout(0.5))

# 10: +Dense

model.add(Dense(1, activation = 'sigmoid'))



#     +Sigmoid

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
# use the generator to train the model (analogue of the fit method)

# 1 epoch of training on a CPU will take 4-6 minutes. The GPU is an ~order of magnitude faster.



model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=20 , #try different number of epochs: 10, 15, 20; check the loss and accuracy;

    validation_data=val_generator,

    validation_steps=nb_validation_samples // batch_size)
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Accuracy on test data: %.2f%%" % (scores[1]*100))
# First, download the weights of the VGG16 network trained on the ImageNet dataset:



vgg16_net = VGG16(weights='imagenet', 

                  include_top=False,      # we take only the "convolution" part, the last layers we add ourselves

                  input_shape=(150, 150, 3))

vgg16_net.trainable = False               # clearly prescribe that we do NOT overload the network.

                                          # Weights VGG16 in the process of learning will remain unchanged!



vgg16_net.summary()                       # pay attention to the number of trained and untrained parameters
# add layers to VGG16:



model = Sequential()

model.add(vgg16_net)



# + flattening

model.add(Flatten())

# + dense connected layer with 256 neurons

model.add(Dense(256, activation = 'relu'))

# + ReLu

# + Dropout

model.add(Dropout(0.5))

# + full layer with 1 neuron

model.add(Dense(1, activation = 'sigmoid'))

# + sigmoid



model.summary()
model.compile(loss='binary_crossentropy',

              optimizer=Adam(lr=1e-5), 

              metrics=['accuracy'])
# We also use the generator to train the model (similar to the fit method)

# Without using a GPU, learning 1 epoch of such a network will take about an hour. Plan your time =)

# If you have access to a GPU, you can try 10-12 epochs - the quality should increase even more.



model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=12,

    validation_data=val_generator,

    validation_steps=nb_validation_samples // batch_size)
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Accuracy on test data: %.2f%%" % (scores[1]*100))