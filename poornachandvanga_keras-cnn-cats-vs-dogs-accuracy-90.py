import warnings

warnings.filterwarnings(action='ignore')

from keras import Sequential

from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten,BatchNormalization

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import os

import matplotlib.pyplot as plt
TRAIN_DIR = '/kaggle/input/cat-and-dog/training_set/training_set'

VALID_DIR = '/kaggle/input/cat-and-dog/test_set/test_set'
WIDTH = 150

HEIGHT = 150

TRAIN_DATA_GEN = ImageDataGenerator(rescale=1.0/255.0,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    height_shift_range=0.1,

                                    width_shift_range=0.1,

                                    rotation_range=30)

TRAIN_DATA_ITER = TRAIN_DATA_GEN.flow_from_directory(TRAIN_DIR,

                                                     color_mode='grayscale',

                                                     target_size=(WIDTH,HEIGHT),

                                                     batch_size=32)

VALID_DATA_GEN = ImageDataGenerator(rescale=1.0/255.0)

VALID_DATA_ITER = VALID_DATA_GEN.flow_from_directory(VALID_DIR,

                                                     color_mode='grayscale',

                                                     target_size=(WIDTH,HEIGHT),

                                                     batch_size=32)
first_batch = TRAIN_DATA_ITER.next()

i = 0

_,ax = plt.subplots(1,6,figsize=(20,2))

for image,label in zip(first_batch[0][:6],first_batch[1][:6]):

  ax[i].imshow(image.reshape(WIDTH,HEIGHT),cmap='gray')

  ax[i].set_title(str(label))

  ax[i].set_xticks([])

  ax[i].set_yticks([])

  i+=1
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(150,150,1),activation='relu',padding='same'))

model.add(Conv2D(32,(3,3),input_shape=(150,150,1),activation='relu',padding='same'))

model.add(MaxPool2D(strides=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D(strides=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D(strides=(2,2)))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))

model.compile(loss=categorical_crossentropy,

              optimizer = Adam(lr=0.0003),

              metrics=['acc'])

model.summary()
early_stopping = EarlyStopping(monitor='val_loss',

                               patience=10,

                               restore_best_weights=True)

steps_per_epoch = TRAIN_DATA_ITER.n//TRAIN_DATA_ITER.batch_size

validation_steps = VALID_DATA_ITER.n//VALID_DATA_ITER.batch_size

history = model.fit_generator(TRAIN_DATA_ITER,

                              steps_per_epoch=steps_per_epoch,

                              epochs=100,

                              validation_data=VALID_DATA_ITER,

                              validation_steps=validation_steps,

                              shuffle=True,

                              callbacks=[early_stopping])
plt.plot(history.history['acc'],label='training_accuracy')

plt.plot(history.history['val_acc'],label='validation_accuracy')

plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title("Accuracy")
plt.plot(history.history['loss'],label='training_loss')

plt.plot(history.history['val_loss'],label='validation_loss')

plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title("Loss")