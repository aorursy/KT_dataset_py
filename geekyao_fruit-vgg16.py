#VGG16 Architectures



from keras.layers import Input, Conv2D, MaxPooling2D

from keras.layers import Dense, Flatten

from keras.models import Model



_input = Input((224,224,1)) 



conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)

conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)

pool1  = MaxPooling2D((2, 2))(conv2)



conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)

conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)

pool2  = MaxPooling2D((2, 2))(conv4)



conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)

conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)

conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)

pool3  = MaxPooling2D((2, 2))(conv7)



conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)

conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)

conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)

pool4  = MaxPooling2D((2, 2))(conv10)



conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)

conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)

conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)

pool5  = MaxPooling2D((2, 2))(conv13)



flat   = Flatten()(pool5)

dense1 = Dense(4096, activation="relu")(flat)

dense2 = Dense(4096, activation="relu")(dense1)

output = Dense(1000, activation="softmax")(dense2)



vgg16_model  = Model(inputs=_input, outputs=output)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/fruits/fruits/fruits"))



# Any results you write to the current directory are saved as output.
#Import Libraries



from __future__ import print_function, division

from builtins import range, input



from keras.layers import Input, Lambda, Dense, Flatten

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator



from sklearn.metrics import confusion_matrix

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from glob import glob

import time
# Explore Data

train_path = '../input/fruits/fruits/fruits/Training'

valid_path = '../input/fruits/fruits/fruits/Test'



# useful for getting number of files

image_files = glob(train_path + '/*/*.jp*g')

valid_image_files = glob(valid_path + '/*/*.jp*g')

print("Number of Images for Training: ",len(image_files))

print("Number of Images for validating: ",len(glob(valid_path + '/*/*.jp*g')))

# useful for getting number of classes

folders = glob(train_path + '/*')

print("Number of classes: ",len(folders))
# re-size all the images to 100x100

IMAGE_SIZE = [100, 100] 



# add preprocessing layer to the front of VGG

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# don't train existing weights

for layer in vgg.layers:

    layer.trainable = False



# our layers

x = Flatten()(vgg.output)

# x = Dense(1000, activation='relu')(x)

prediction = Dense(len(folders), activation='softmax')(x)





# create a model object

model = Model(inputs=vgg.input, outputs=prediction)



# view the structure of the model

model.summary()
# Loss and Optimization functions

model.compile(

  loss='categorical_crossentropy',

  optimizer='rmsprop',

  metrics=['accuracy']

)



# create an instance of ImageDataGenerator

gen = ImageDataGenerator(

  rotation_range=20,

  width_shift_range=0.1,

  height_shift_range=0.1,

  shear_range=0.1,

  zoom_range=0.2,

  horizontal_flip=True,

  vertical_flip=True,

  preprocessing_function=preprocess_input

)
# training config:



start = time.time()

epochs = 50

batch_size = 64



# create generators

train_generator = gen.flow_from_directory(

  train_path,

  target_size=IMAGE_SIZE,

  shuffle=True,

  batch_size=batch_size,

)

valid_generator = gen.flow_from_directory(

  valid_path,

  target_size=IMAGE_SIZE,

  shuffle=True,

  batch_size=batch_size,

)



# fit the model

r = model.fit_generator(

  train_generator,

  validation_data=valid_generator,

  epochs=epochs,

  steps_per_epoch=len(image_files) // batch_size,

  validation_steps=len(valid_image_files) // batch_size,

)



end = time.time()

totalTime = end - start

print("Total time is: %.2f seconds" % (totalTime))
# plot Loss and Accuracies



# loss

plt.plot(r.history['loss'], label='train loss')

plt.plot(r.history['val_loss'], label='val loss')

plt.legend()

plt.show()



# accuracies

plt.plot(r.history['acc'], label='train acc')

plt.plot(r.history['val_acc'], label='val acc')

plt.legend()

plt.show()