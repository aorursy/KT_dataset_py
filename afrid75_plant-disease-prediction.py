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
import tensorflow as tf

import glob

import matplotlib.pyplot as plt

import os

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(1./255)



EPOCHS=20

BATCH_SIZE = 50

IMG_HEIGHT = 128

IMG_WIDTH = 128


# Initializing the CNN

classifier = Sequential()



# Convolution Step 1

classifier.add(Convolution2D(96, 5, strides = (4, 4), padding = 'valid', input_shape=(128, 128, 3), activation = 'relu'))



# Max Pooling Step 1

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

# classifier.add(BatchNormalization())



# Convolution Step 2

classifier.add(Convolution2D(256, 5, strides = (1, 1), padding='valid', activation = 'relu'))



# Max Pooling Step 2

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))

# classifier.add(BatchNormalization())



# Convolution Step 3

classifier.add(Convolution2D(384, 2, strides = (1, 1), padding='valid', activation = 'relu'))

# classifier.add(BatchNormalization())



# Convolution Step 4

classifier.add(Convolution2D(384, 2, strides = (1, 1), padding='valid', activation = 'relu'))

# classifier.add(BatchNormalization())



# Convolution Step 5

classifier.add(Convolution2D(256, 2, strides=(1,1), padding='valid', activation = 'relu'))



# Max Pooling Step 3

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

# classifier.add(BatchNormalization())



# Flattening Step

classifier.add(Flatten())



# Full Connection Step

classifier.add(Dense(units = 4096, activation = 'relu'))

classifier.add(Dropout(0.4))

# classifier.add(BatchNormalization())

classifier.add(Dense(units = 4096, activation = 'relu'))

classifier.add(Dropout(0.4))

# classifier.add(BatchNormalization())

classifier.add(Dense(units = 1000, activation = 'relu'))

classifier.add(Dropout(0.2))

# classifier.add(BatchNormalization())

classifier.add(Dense(units = 38, activation = 'softmax'))

# classifier.summary()


train_data_gen = image_generator.flow_from_directory(directory=r'../input/plant-disease/dataset/train',

                                                     batch_size=BATCH_SIZE,

                                                     shuffle=True,

                                                     class_mode="sparse",

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))





validation_data_gen = image_generator.flow_from_directory(directory=r'../input/plant-disease/dataset/test',

                                                     batch_size=BATCH_SIZE,

                                                     shuffle=True,

                                                    class_mode="sparse",

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_path = "../output/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,

                                                 save_weights_only=True,

                                                 verbose=1)
classifier.fit(train_data_gen, validation_data=validation_data_gen, epochs=EPOCHS, callbacks=[cp_callback])
plt.plot(classifier.history.__dict__['history']['loss'], color='b', label="train_loss")

plt.plot(classifier.history.__dict__['history']['val_loss'], color='r', label="val_loss")

plt.title("Loss")

plt.legend()