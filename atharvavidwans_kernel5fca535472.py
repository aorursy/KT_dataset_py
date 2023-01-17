# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras # deeplearning

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop #optimizer for CNN model

import matplotlib.pyplot as plt #plotting data

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print('Training Data avaiable is')

print(int(len(os.listdir('../input/skin-cancer-malignant-vs-benign/data/train/benign')))+int(len(os.listdir('../input/skin-cancer-malignant-vs-benign/data/train/malignant'))))



print('Testing Data avaiable is')

print(int(len(os.listdir('../input/skin-cancer-malignant-vs-benign/data/test/benign')))+int(len(os.listdir('../input/skin-cancer-malignant-vs-benign/data/test/malignant'))))





#Model for Convolutional Neural Network with pooling



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(228,228,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid'),

])

model.summary()
#Compile the model



model.compile(

    loss='binary_crossentropy',

    optimizer = RMSprop(lr=0.0001),

    metrics=['acc']

)
#Using Image Generator 



path_training = '../input/skin-cancer-malignant-vs-benign/data/train/'

path_validation = '../input/skin-cancer-malignant-vs-benign/data/test/'

train_image_data_gen=ImageDataGenerator(

    rescale=1./255,

    rotation_range=90,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest' 

)



train_image_generator=train_image_data_gen.flow_from_directory(

    path_training ,

    batch_size = 10,

    class_mode = 'binary' ,

    target_size = (228,228) ,

)



validation_image_data_gen=ImageDataGenerator(

    rescale=1./255

)

validation_image_generator= validation_image_data_gen.flow_from_directory(

    path_validation ,

    class_mode = 'binary' ,

    target_size = (228,228) ,

    batch_size = 10

)
#Actual Fitting of the data to the model



data = model.fit_generator(

    train_image_generator,

    validation_data=validation_image_generator,

    epochs=10,

    verbose=2

)
#Plotting the data of Accuracy and Loss using Matplotlib



%matplotlib inline



acc=data.history['acc']

val_acc=data.history['val_acc']

loss=data.history['loss']

val_loss=data.history['val_loss']



epochs=range(len(acc)) 



plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.figure()



plt.plot(epochs, loss ,'r','Training Loss')

plt.plot(epochs, val_loss, 'b','Validation Loss')

plt.title('Training and Validation Loss')

plt.figure()