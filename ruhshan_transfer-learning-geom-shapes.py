# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/geom_shapes/shapes"))

# Any results you write to the current directory are saved as output.
from tensorflow.python.keras.applications import ResNet50 # pretrained resnet50 model
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 3
resnet_weights_path = '../input/geom_shapes/shapes/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential() 
my_new_model.add(ResNet50(include_top=False,
#                           pooling='max',
                          pooling='avg', 
                          weights=resnet_weights_path))
my_new_model.add(Flatten())
my_new_model.add(Dense(300, activation='relu'))
my_new_model.add(Dense(num_classes, activation='softmax'))
my_new_model.layers[0].trainable = False # use pretrained resnet50 model
from tensorflow.python.keras import optimizers
optmizer_adam = optimizers.Adam(lr=0.01,)
my_new_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator()
train_data = '../input/geom_shapes/shapes/train' # 2726 images
valid_data = '../input/geom_shapes/shapes/valid' # 300 images

train_generator = data_generator.flow_from_directory(
       directory = train_data,
       target_size=(image_size, image_size),
       batch_size= 29, 
       class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
       directory = valid_data,
       target_size=(image_size, image_size),
       class_mode='categorical')

my_new_model.fit_generator(
       train_generator,
       steps_per_epoch=94, # trainingsamples/batchsize 
       validation_data=validation_generator,
       validation_steps=1,
        epochs=4)
my_new_model.summary()