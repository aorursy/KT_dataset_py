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
os.listdir('../input/train')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')
train_generator=train_datagen.flow_from_directory("../input/train/",batch_size=20,target_size=(256,256),

                                                  class_mode='categorical')
from tensorflow.keras import layers

from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
pre_trained_model=InceptionV3(input_shape=(256, 256, 3),

                             include_top= False,

                             weights= 'imagenet')
for layer in pre_trained_model.layers:

    layer.trainable=False
last_layer=pre_trained_model.get_layer('mixed7')

print('last layer output shape: ',last_layer.output_shape)

last_output=last_layer.output
from tensorflow.keras.optimizers import RMSprop



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

# Add a final sigmoid layer for classification

x = layers.Dense  (12, activation='softmax')(x)           



model = Model( pre_trained_model.input, x) 



model.compile(optimizer = RMSprop(lr=0.0001), 

              loss = 'categorical_crossentropy', 

              metrics = ['acc'])
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=50)
model.save('modelp.h5')