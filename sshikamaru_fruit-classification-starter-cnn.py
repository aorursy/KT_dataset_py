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
TRAIN_DIR = '/kaggle/input/fruit-recognition/train/train'

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers as L

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



import matplotlib.pyplot as plt
img_datagen = ImageDataGenerator(rescale=1./255,

                                vertical_flip=True,

                                horizontal_flip=True,

                                rotation_range=40,

                                width_shift_range=0.2,

                                height_shift_range=0.2,

                                zoom_range=0.1,

                                validation_split=0.2)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = img_datagen.flow_from_directory(TRAIN_DIR,

                                                 shuffle=True,

                                                 batch_size=32,

                                                 subset='training',

                                                 target_size=(100, 100))



valid_generator = img_datagen.flow_from_directory(TRAIN_DIR,

                                                 shuffle=True,

                                                 batch_size=16,

                                                 subset='validation',

                                                 target_size=(100, 100))
model = Sequential()



# model.add(L.Conv2D(64, (5, 5), activation='relu', padding='Same', input_shape=(100, 100, 3)))

# model.add(L.Conv2D(64, (5, 5), activation='relu', padding='Same'))

# model.add(L.MaxPool2D((2, 2)))

# model.add(L.Dropout(0.25))



# model.add(L.Conv2D(128, (3, 3), activation='relu', padding='Same'))

# model.add(L.Conv2D(128, (3, 3), activation='relu', padding='Same'))

# model.add(L.MaxPool2D((2, 2), strides=(2, 2)))

# model.add(L.Dropout(0.25))



model.add(keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',

                                                                  include_top=False,

                                                                  input_shape=(100, 100, 3)))

model.add(L.Flatten())

model.add(L.Dense(256, activation='relu'))

model.add(L.Dropout(0.5))

model.add(L.Dense(33, activation='softmax'))



model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('val_accuracy') >= 0.997):

            print("\nReached 99.7% accuracy so cancelling training!")

            self.model.stop_training = True



early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max',

                                        restore_best_weights=True)

            

history = model.fit(train_generator, validation_data=valid_generator,

                   steps_per_epoch=train_generator.n//train_generator.batch_size,

                   validation_steps=valid_generator.n//valid_generator.batch_size,

                    callbacks=[early],

                   epochs=10)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.show()
TEST_DIR = '../input/fruit-recognition/test/test'

def number(num):

    if len(num) == 4:

        return num

    elif len(num) == 3:

        return '0'+num

    elif len(num) == 2:

        return '00'+num

    else:

        return '000'+num