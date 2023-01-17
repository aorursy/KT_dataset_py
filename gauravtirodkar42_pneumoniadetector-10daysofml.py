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
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
train_dir = '../input/chest-xray-pneumonia/chest_xray/train'
test_dir = '../input/chest-xray-pneumonia/chest_xray/test'
val_dir = '../input/chest-xray-pneumonia/chest_xray/val'

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 16
img_width, img_height = 150,150
input_shape = (img_width,img_height,3)
epochs = 20
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = 'binary')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape=input_shape,activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(loss='binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 326,
    epochs = epochs,
    validation_data = val_generator,
    validation_steps = 1)
scores = model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*180))
