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
import pathlib

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
train_dir='/kaggle/input/intel-image-classification/seg_train/seg_train/'

train_datagen = ImageDataGenerator( rescale = 1.0/255,

                                          width_shift_range=0.2,

                                          height_shift_range=0.2,

                                          zoom_range=0.2,

                                          vertical_flip=True,

                                          fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=32,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))

test_dir='/kaggle/input/intel-image-classification/seg_test/seg_test/'

test_datagen=ImageDataGenerator(rescale=1/255,

                               width_shift_range=0.2,

                               height_shift_range=0.2,zoom_range=0.2,vertical_flip=True,

                               fill_mode='nearest')

test_generator=test_datagen.flow_from_directory(test_dir,batch_size=32,

                                               class_mode='categorical',

                                               target_size=(150,150))
from tensorflow.keras.applications import InceptionV3
pre_trained_model=InceptionV3(include_top=False,input_shape=(150,150,3))

for layer in pre_trained_model.layers:

    layer.trainable = False
last_layer=pre_trained_model.get_layer('mixed9')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output                                       
from tensorflow.keras.optimizers import RMSprop,Adam

from tensorflow import keras

from keras.callbacks import ReduceLROnPlateau
x = tf.keras.layers.Flatten()(last_output)

x = tf.keras.layers.Dense(units = 1024, activation = tf.nn.relu)(x)

x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense  (6, activation = tf.nn.softmax)(x)



model = tf.keras.Model( pre_trained_model.input, x)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=1,

                                            verbose=1,

                                            factor=0.25,

                                            min_lr=0.000003)



model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0001), 

              loss = 'categorical_crossentropy', 

              metrics = ['accuracy'])
model.summary()
history = model.fit(

            train_generator,

            verbose=1,

            validation_data = test_generator,

            epochs = 10,

    callbacks=[learning_rate_reduction])

import matplotlib.pyplot as plt
%matplotlib inline

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()