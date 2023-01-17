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
import matplotlib.image as img

import matplotlib.pyplot as plt
sample = img.imread('/kaggle/input/gravity-spy-gravitational-waves/train/train/Power_Line/H1_9wlnDo4Vtg_spectrogram_4.0.png')
plt.imshow(sample)

print(sample.shape)
from keras_preprocessing.image import ImageDataGenerator
TRAINING_DIR = '/kaggle/input/gravity-spy-gravitational-waves/train/train/'

training_datagen = ImageDataGenerator(rescale = 1./255)



VALIDATION_DIR = '/kaggle/input/gravity-spy-gravitational-waves/validation/validation/'

validation_datagen = ImageDataGenerator(rescale = 1./255)
sample.shape[0]
train_generator = training_datagen.flow_from_directory(

	TRAINING_DIR,

	target_size=(sample.shape[0],sample.shape[1]),

    batch_size = 32,

	class_mode='categorical',

  shuffle=True

)



validation_generator = validation_datagen.flow_from_directory(

	VALIDATION_DIR,

	target_size=(sample.shape[0],sample.shape[1]),

    batch_size = 32,

	class_mode='categorical',

  shuffle=True

)
train_generator.class_indices
import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image
model = tf.keras.models.Sequential([



    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(sample.shape[0],sample.shape[1],3)),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),



    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(22, activation='softmax')

])



model.summary()



model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
callbacks = tf.keras.callbacks.EarlyStopping(patience=2)



history = model.fit_generator(train_generator, epochs=50, 

                              validation_data = validation_generator, validation_steps = 32,

                              verbose = 1, steps_per_epoch = 32, callbacks=[callbacks])
plt.grid()



plt.plot(history.history['accuracy'], label = 'Train')

plt.plot(history.history['val_accuracy'], label = 'Val')

plt.title('Accuracy')

plt.legend()

plt.show()



plt.grid()

plt.figure()

plt.plot(history.history['loss'], label = 'Train')

plt.plot(history.history['val_loss'], label = 'Val')

plt.title('Loss')

plt.legend()
TEST_DIR = '/kaggle/input/gravity-spy-gravitational-waves/test/test/'

test_datagen = ImageDataGenerator(rescale = 1./255)



test_generator = test_datagen.flow_from_directory(

	TEST_DIR,

	target_size=(sample.shape[0],sample.shape[1]),

    batch_size = 1,

	class_mode='categorical',

  shuffle=False

)
prediction = model.predict_generator(test_generator, steps = 4720 , verbose = 1)