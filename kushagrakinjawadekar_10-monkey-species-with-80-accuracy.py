# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/10-monkey-species'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing import image

from tensorflow.keras.callbacks import ReduceLROnPlateau

import tensorflow as tf

from pathlib import Path

import os

import random

from shutil import copyfile

import cv2
train_dir = Path('../input/10-monkey-species/training/training')

test_dir = Path('../input/10-monkey-species/validation/validation')
import matplotlib.pyplot as plt

%matplotlib inline

img = image.load_img('/kaggle/input/10-monkey-species/training/training/n0/n0140.jpg')

plt.imshow(img)

plt.show()
#label info

cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']

labels = pd.read_csv("/kaggle/input/10-monkey-species/monkey_labels.txt", names=cols, skiprows=1)

labels
labels = labels['Common Name']

labels
LR = 1e-3

height=150

width=150

channels = 3

batch_size = 64

num_classes = 10

epochs = 200

data_augmentation = True

num_predictions = 20

seed = 1337

data_augmentation = True

num_predictions = 20





train_datagen = ImageDataGenerator(rescale = 1.0/255.0,

                                  rotation_range= 20,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')                                 



train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=batch_size,

                                                    seed=seed,

                                                    class_mode='categorical',

                                                    shuffle=True,

                                                    target_size=(height, width))

                                                  





validation_datagen = ImageDataGenerator(rescale = 1.0/255.0)



validation_generator =  validation_datagen.flow_from_directory(test_dir,

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    target_size=(height, width))

                                                    

                                    



train_num = train_generator.samples

validation_num = validation_generator.samples 
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape = (150,150,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape = (150,150,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape = (150,150,3)),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape = (150,150,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes,activation='softmax')

         

])
from keras.callbacks import ModelCheckpoint, EarlyStopping





filepath=str(os.getcwd()+"/model.h5f")

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# = EarlyStopping(monitor='val_acc', patience=15)

callbacks_list = [checkpoint]#, stopper]
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])

model.summary()
history = model.fit(train_generator,

                   steps_per_epoch= train_num // batch_size,

                   epochs=epochs,

                   verbose=1,

                   validation_data = validation_generator,

                   validation_steps= validation_num // batch_size, 

                   callbacks = callbacks_list)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'red', label='Training acc')

plt.plot(epochs, val_acc, 'blue', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'red', label='Training loss')

plt.plot(epochs, val_loss, 'blue', label='Validation loss')



plt.legend()



plt.show()