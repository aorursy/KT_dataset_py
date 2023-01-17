import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import random

import os

print('setup is complete')
test_dataset = pd.read_csv("../input/digit-recognizer/test.csv")

train_dataset = pd.read_csv("../input/digit-recognizer/train.csv")

print('datasets are ready')
plt.figure(figsize=(18, 8))

sample_image = train_dataset.sample(18).reset_index(drop=True)

for index, row in sample_image.iterrows():

    label = row['label']

    image_pixels = row.drop('label')

    plt.subplot(3, 6, index+1)

    plt.imshow(image_pixels.values.reshape(28,28), cmap=plt.cm.gray)

    plt.title(label)

plt.tight_layout()
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



x = train_dataset.drop(columns=['label']).values.reshape(train_dataset.shape[0],28,28,1)

y = to_categorical(train_dataset['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
from keras.preprocessing.image import ImageDataGenerator

batch_size=32

train_datagen = ImageDataGenerator(

    rotation_range=10,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_datagen.fit(x_train)

train_generator = train_datagen.flow(

    x_train,

    y_train,

    batch_size=batch_size

)



validation_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(x_test)

validation_generator = validation_datagen.flow(

    x_test,

    y_test

    

)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D



model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



callbacks = [

    EarlyStopping(patience=10, verbose=1),

    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),

    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)

]
model.fit_generator(

    train_generator, 

    steps_per_epoch=len(x_train) // batch_size, 

    validation_data=validation_generator,

    validation_steps=len(x_test) // batch_size,

    epochs=150,

    callbacks=callbacks

)
test_digit_data = test_dataset.values.reshape(test_dataset.shape[0],28,28,1).astype("float32") / 255

predictions = model.predict(test_digit_data)

results = np.argmax(predictions, axis = 1) 
plt.figure(figsize=(18, 12))

sample_test = test_dataset.head(30)

for index, image_pixels in sample_test.iterrows():

    label = results[index]

    plt.subplot(5, 6, index+1)

    plt.imshow(image_pixels.values.reshape(28,28), cmap=plt.cm.gray)

    plt.title(label)

plt.tight_layout()