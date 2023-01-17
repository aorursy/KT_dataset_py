# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from keras import layers, models

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam, Adamax

from keras.regularizers import l2

import glob

import cv2
path="../input/"



train_path=path+"train/train/"

test_path=path+"test/test/"
files = sorted(glob.glob(train_path + '*.jpg'))

files[:3]

train_raw = np.array([cv2.imread(image)

                      for image in files], dtype='int32')
test_files = sorted(glob.glob(test_path + '*.jpg'))

test_files[:3]

test_raw = np.array([cv2.imread(image)

                      for image in test_files], dtype='int32')
train_images = train_raw/255

test_images = test_raw/255
train_set = pd.read_csv(path + 'train.csv')

train_labels = train_set['has_cactus']

train_labels = to_categorical(train_labels)
model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(32, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(32, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_uniform'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dense(128, kernel_regularizer=l2(5e-4)))

model.add(layers.Dense(64, kernel_regularizer=l2(5e-4)))

model.add(layers.Dense(64, kernel_regularizer=l2(5e-4)))

model.add(layers.Dense(32, kernel_regularizer=l2(5e-4)))

model.add(layers.Flatten())

model.add(layers.Dense(256))

model.add(layers.Dense(256))

model.add(layers.Dense(32))

model.add(layers.Dense(2, activation='softmax'))
model.summary()
cbl = [

    EarlyStopping(

        monitor='val_loss',

        patience=10,

    ),

    ModelCheckpoint(

        filepath='best_model.h5',

        monitor='val_loss',

        save_best_only=True,

    ),

    ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.1,

        patience=10

    )

]
model.compile(optimizer=Adam(),

              loss='binary_crossentropy',

              metrics=['accuracy', ],

              )
%%time



history = model.fit(

    train_images,

    train_labels,

    validation_split=0.01,

    epochs=200,

    batch_size=32,

    callbacks=cbl

)
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
predicted_data = model.predict(test_images)

predicted_data
test_data = pd.read_csv(path + 'sample_submission.csv')

test_data['has_cactus'] = predicted_data
test_data.to_csv('sample_submission.csv', index=False)
pd.read_csv('sample_submission.csv').head(42)
test_data.query("0.4 <= has_cactus <= 0.6").shape