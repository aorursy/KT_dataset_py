# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)
base_model.trainable = False
# Create inputs with correct shape
inputs = base_model.input

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.Flatten()(base_model.output)

# Add final dense layer
outputs = keras.layers.Dense(6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer='rmsprop')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator( rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  rescale=1./255,  
  preprocessing_function=keras.applications.vgg16.preprocess_input)
# load and iterate training dataset
train_path='../input/fruits-fresh-and-rotten-for-classification/dataset/train'  #on a travaillé sur le notebook local de Kaggle
test_path='../input/fruits-fresh-and-rotten-for-classification/dataset/test'

train_it = datagen.flow_from_directory(train_path, 
                                       target_size=[224,224], 
                                       color_mode='rgb', 
                                       class_mode="categorical",
                                       batch_size = 32)
# load and iterate test dataset
test_it = datagen.flow_from_directory(test_path, 
                                      target_size=[224,224], 
                                      color_mode='rgb', 
                                      class_mode="categorical",
                                    batch_size = 32)
model.fit(train_it,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=10)
