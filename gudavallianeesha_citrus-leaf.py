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
# Directory with our training blackspot pictures
train_blackspot_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/train/blackspot')


# Directory with our training canker pictures
train_canker_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/train/canker')


train_greening_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/train/greening')

train_healthy_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/train/healthy')


# Directory with our validation  pictures
validation_blackspot_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/validation/blackspot')

# Directory with our validation  pictures
validation_canker_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/validation/canker')


validation_greening_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/validation/greening')

validation_healthy_dir = os.path.join('/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/validation/healthy')





train_blackspot_names = os.listdir(train_blackspot_dir)
print(train_blackspot_names[:10])

train_canker_names = os.listdir(train_canker_dir)
print(train_canker_names[:10])


train_greening_names = os.listdir(train_greening_dir)
print(train_greening_names[:10])


train_healthy_names = os.listdir(train_healthy_dir)
print(train_healthy_names[:10])


validation_blackspot_hames = os.listdir(validation_blackspot_dir)
print(validation_blackspot_hames[:10])

validation_canker_names = os.listdir(validation_canker_dir)
print(validation_canker_names[:10])


validation_greening_names = os.listdir(validation_greening_dir)
print(validation_greening_names[:10])


validation_healthy_names = os.listdir(validation_healthy_dir)
print(validation_healthy_names[:10])


import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),
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
    tf.keras.layers.Dense(4, activation='sigmoid')
])
model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(
              optimizer=RMSprop(lr=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. /255,rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, zoom_range=0.4,vertical_flip=True)
validation_datagen = ImageDataGenerator(rescale=1/255, shear_range=0.2)

train_generator = train_datagen.flow_from_directory(
        '/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/train',  
        target_size=(256, 256), 
        batch_size=10
        )

validation_generator = validation_datagen.flow_from_directory(
        '/kaggle/input/citrus-leaves-prepared/citrus_leaves_prepared/validation', 
        target_size=(256, 256),  
        batch_size=32,
        )

epochs=20
history = model.fit_generator(
      train_generator,  
      epochs=epochs,
      verbose=1,
      validation_data = validation_generator
)




loss = history.history["loss"]
acc = history.history["accuracy"]
val_loss = history.history["val_loss"]
val_acc = history.history["val_accuracy"]
import matplotlib.pyplot as plt
epoch = range(epochs)
plt.plot(loss,epoch,label = "train_loss")
plt.plot(val_loss,epoch,label = "val loss")
plt.legend()
plt.plot(acc,epoch,label = "train_accuracy")
plt.plot(val_acc,epoch,label = "val accuracy")
plt.legend()