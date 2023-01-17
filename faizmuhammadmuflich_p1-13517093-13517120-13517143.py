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
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
print(os.listdir("../input"))
FAST_RUN = False
IMAGE_WIDTH=256
IMAGE_HEIGHT=256
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
VALID_FRACTION = 0.2
BATCH_SIZE = 1
EPOCHS = 50

IMAGE_WIDTH = IMAGE_HEIGHT = 256

TRAIN_DIR = '/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/train/'
# creating df with train labels
directories = os.listdir(TRAIN_DIR)
train_labels = []
train_filenames = []
for directory in directories:
    filenames = os.listdir(TRAIN_DIR + directory)
    for i, index in enumerate(filenames):
        filenames[i] = directory + "/" + filenames[i]
    train_filenames = train_filenames + filenames
    for filename in filenames:
        if (directory == "1"):
            train_labels.append('cloudy')
        elif (directory == "2"):
            train_labels.append('rain')
        elif (directory == "3"):
            train_labels.append('shine')
        else: 
            train_labels.append('sunrise')


train_df = pd.DataFrame({
    'id': train_filenames,
    'label': train_labels
})

print(train_df)
import tensorflow as tf

IMAGE_WIDTH = IMAGE_HEIGHT = 256
VALID_FRACTION = 0.2

train_df, valid_df = train_test_split(train_df, test_size=VALID_FRACTION)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(    
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255.,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    TRAIN_DIR, 
    x_col='id',
    y_col='label',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

valid_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df, 
    TRAIN_DIR, 
    x_col='id',
    y_col='label',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(256, 256, 3)),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2), 
#     tf.keras.layers.Flatten(), 
#     tf.keras.layers.Dense(512, activation='relu'), 
#     tf.keras.layers.Dense(512, activation='relu'), 
#     tf.keras.layers.Dense(4, activation='softmax'),
# ])

# model.summary()

# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
#     loss="categorical_crossentropy",
#     metrics = ['accuracy'])

# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#     mode='min',
#     restore_best_weights=True, 
#     verbose=1,
#     patience=5)
# percobaan

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, ZeroPadding2D

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss="categorical_crossentropy",
    metrics = ['accuracy'])
# tf.keras.utils.plot_model(model)
TRAIN_SIZE = len(train_filenames)
BATCH_SIZE = 20
# history = model.fit_generator(train_generator,
#     validation_data=valid_generator,
#     steps_per_epoch=round(TRAIN_SIZE*(1.-VALID_FRACTION)/BATCH_SIZE),
#     validation_steps=round(TRAIN_SIZE*VALID_FRACTION/BATCH_SIZE),
#     epochs=EPOCHS,
#     callbacks=[es],
#     verbose=1)

history = model.fit (train_generator,
          validation_data=valid_generator,
          epochs=50)

TEST_DIR = '/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/test'


# preparing testing data
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({
    'id': test_filenames
})
TEST_SIZE = len([name for name in test_filenames])
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.)

test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    TEST_DIR, 
    x_col='id',
    y_col=None,
    class_mode=None,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False
)

yhat = model.predict_generator(test_generator, steps=np.ceil(TEST_SIZE/BATCH_SIZE))
yhat = [np.argmax(y) for y in yhat]

test_df['label'] = yhat

test_df.to_csv('submission.csv', index=False)
test_df.head()
