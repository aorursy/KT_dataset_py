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
import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping

import random

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data = '/kaggle/input/dogs-cats-images/dataset/training_set'

train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_data,

                                                    batch_size=10,

                                                    class_mode='binary',

                                                    target_size=(150, 150))



valid_data = '/kaggle/input/dogs-cats-images/dataset/test_set'

validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(valid_data,

                                                              batch_size=10,

                                                              class_mode='binary',

                                                              target_size=(150, 150))

class_names = dict([(value, key) for key, value in train_generator.class_indices.items()]) 
plt.figure(figsize=(10, 10))

images,labels = next(train_generator)

print(images.shape)

for i in range(9):

    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(images[i])

    plt.title(class_names[labels[i]])

    plt.axis("off")
early_stop = EarlyStopping(monitor='val_loss',patience=4)

checkpoint_filepath = '/kaggle/working/'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_filepath,

    save_weights_only=True,

    monitor='val_acc',

    mode='max',

    save_best_only=True)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

model.summary()


history = model.fit_generator(

    train_generator,

    epochs=10,

    verbose=1,

    validation_data=validation_generator,

    callbacks=[model_checkpoint_callback]

)
base_model = tf.keras.applications.MobileNetV2(input_shape = (150,150,3),

                                            include_top=False,

                                           weights='imagenet')

for layer in base_model.layers:

    layer.trainable = False

model = tf.keras.Sequential([

                          base_model,

                          keras.layers.GlobalAveragePooling2D(),

                          keras.layers.Dense(1, activation='sigmoid')])



model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

model.summary()
history = model.fit_generator(

    train_generator,

    epochs=10,

    verbose=1,

    validation_data=validation_generator,

    callbacks=[early_stop,model_checkpoint_callback]

)