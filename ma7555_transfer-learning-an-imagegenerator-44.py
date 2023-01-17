# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras

import os

from tqdm import tqdm

from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Configs

batch_size = 2048

nepochs = 10

img_width = 300

img_height = 300
root_dir = r'/kaggle/input/cat-breeds-dataset/images/'



train_datagen = ImageDataGenerator(rescale=1./255,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    validation_split=0.2) # set validation split



train_generator = train_datagen.flow_from_directory(root_dir,

                                                    target_size=(img_width, img_height),

                                                    batch_size=256,

                                                    class_mode='categorical',

                                                    subset='training') # set as training data



validation_generator = train_datagen.flow_from_directory(root_dir, # same directory as training data

                                                            target_size=(img_width, img_height),

                                                            batch_size=256,

                                                            class_mode='categorical',

                                                            subset='validation') # set as validation data
model_base = keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', 

                                                                      include_top=False, 

                                                                      input_shape=(img_width,img_height,3))



model = keras.models.Sequential()

for layer in model_base.layers:

    layer.trainable = False



model.add(model_base)

#model.add(keras.layers.GlobalAveragePooling2D())

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(67, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_generator,

                                steps_per_epoch = train_generator.samples // batch_size,

                                validation_data = validation_generator, 

                                validation_steps = validation_generator.samples // batch_size,

                                epochs = nepochs)