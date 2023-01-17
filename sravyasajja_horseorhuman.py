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

# Directory with our training horse pictures
train_horse_dir = os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/train/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/train/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/train/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('/kaggle/input/horses-or-humans-dataset/horse-or-human/train/humans')
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255,rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, zoom_range=0.4,vertical_flip=True)
validation_datagen = ImageDataGenerator(rescale=1/255, shear_range=0.2)

train_generator = train_datagen.flow_from_directory(
        '/kaggle/input/horses-or-humans-dataset/horse-or-human/train',  
        target_size=(300, 300), 
        batch_size=128,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        '/kaggle/input/horses-or-humans-dataset/horse-or-human/validation', 
        target_size=(300, 300),  
        batch_size=32,
        class_mode='binary')
epochs=15
history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=epochs,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
loss = history.history["loss"]
acc = history.history["acc"]
val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]
import matplotlib.pyplot as plt
epoch = range(epochs)
plt.plot(loss,epoch,label = "train loss")
plt.plot(val_loss,epoch,label = "val loss")
plt.legend()
plt.plot(acc,epoch,label = "train acc")
plt.plot(val_acc,epoch,label = "val acc")
plt.legend()
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
