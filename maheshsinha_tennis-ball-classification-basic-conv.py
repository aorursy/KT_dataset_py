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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import cv2
img = cv2.imread('/kaggle/input/ballsemtpytt/emptyballs/train/empty/empty70.jpg')
plt.imshow(img)
# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS

# USE AT LEAST 3 CONVOLUTION LAYERS

model = tf.keras.models.Sequential([

    #First conv layer

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Dropout(0.2),

    #Second conv layer

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    #Third Conv Layer:

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
train_dir = '/kaggle/input/ballsemtpytt/emptyballs/train/'

train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=40, 

                                   width_shift_range=0.2,

                                   height_shift_range=0.2, 

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   fill_mode='nearest',

                                   horizontal_flip=True

                                  )

train_generator = train_datagen.flow_from_directory(train_dir,

                                                   class_mode='binary',

                                                   batch_size = 20,

                                                   target_size = (64, 64))
val_dir = '/kaggle/input/ballsemtpytt/emptyballs/test/'

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

val_generator = val_datagen.flow_from_directory(val_dir,

                                               class_mode='binary',

                                               batch_size=20,

                                               target_size = (64, 64))
hostory = model.fit_generator(train_generator, epochs=450, validation_data=val_generator, verbose=1)
# PLOT LOSS AND ACCURACY

%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=hostory.history['acc']

val_acc=hostory.history['val_acc']

loss=hostory.history['loss']

val_loss=hostory.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.legend(handles=[line1], loc=1)

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")





plt.title('Training and validation loss')



# Desired output. Charts with training and validation metrics. No crash :)