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
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, layers
train_loc='../input/chest-xray-pneumonia/chest_xray/train'
test_loc= '../input/chest-xray-pneumonia/chest_xray/test'
val_loc='../input/chest-xray-pneumonia/chest_xray/val'


input_height=128
input_width=128
batch_size=32
train_ds=tf.keras.preprocessing.image_dataset_from_directory(train_loc, 
                                                             color_mode='grayscale', 
                                                             image_size=(input_height, input_width), 
                                                             batch_size=batch_size)

test_ds=tf.keras.preprocessing.image_dataset_from_directory(test_loc, 
                                                             color_mode='grayscale', 
                                                             image_size=(input_height, input_width), 
                                                             batch_size=batch_size)

val_ds=tf.keras.preprocessing.image_dataset_from_directory(val_loc, 
                                                             color_mode='grayscale', 
                                                             image_size=(input_height, input_width), 
                                                             batch_size=batch_size)
train_ds.take(1)
train_ds.class_names
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        
        plt.subplot(3, 3, i+1)
        plt.imshow(np.squeeze(images[i].numpy().astype('uint8')))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis('off')

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds=train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds=test_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE)
model=Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1./255))

model.add(layers.Conv2D(32,3, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32,3, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32,3, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(2, activation='sigmoid'))


model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)
model.evaluate(test_ds)