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
import keras

import cv2

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16

from keras.models import load_model,model_from_json

from keras.applications.vgg16 import preprocess_input
import tensorflow as tf

import PIL
path = "../input/alien_vs_predator_thumbnails/data" 

train_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True,shear_range = 10,zoom_range = 0.2,preprocessing_function =preprocess_input )

train_generator = train_datagen.flow_from_directory(path+"/train",target_size = (224,224),shuffle = True,class_mode = 'binary')



validation_datagen = ImageDataGenerator(rescale = 1./255,preprocessing_function = preprocess_input)

validation_generator = validation_datagen.flow_from_directory(path + "/validation",target_size = (224,224),shuffle = False,class_mode = 'binary')
conv_base = VGG16(include_top = False,weights = 'imagenet')
for layer in conv_base.layers:

    layer.trainable = False

X = conv_base.output

X = keras.layers.GlobalAveragePooling2D()(X)

X = keras.layers.Dense(128,activation = 'relu')(X)

output_1 = keras.layers.Dense(2,activation = 'softmax')(X)

model = keras.Model(inputs = conv_base.input,outputs = output_1)
model.summary()
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'Adam', metrics = ['accuracy'])
history = model.fit_generator(train_generator,steps_per_epoch = 10,epochs = 10,verbose = 1, validation_data = validation_generator, validation_steps = 10)
model.save_weights("weights.hdf5")
model.evaluate_generator(validation_generator,steps = len(validation_generator),verbose = 1)
from IPython.display import Image
Image(os.path.join(path,"train/predator/127.jpg"))

y_pred = model.predict