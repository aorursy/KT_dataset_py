import os

import zipfile

from tqdm import tqdm

import numpy as np

import cv2

from PIL import Image

from matplotlib import pyplot as plt

%matplotlib inline

## import Kereas and its module for image processing and model building

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
# this is the augmentation configuration we will use for training

# The image values are rescaled to 0-1.

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)

#Number of Batches during each epoch

batch_size = 64

# this is a generator that will read pictures found in

# subfolers of '../input/train/train', and indefinitely generate

# batches of augmented image data

train_generator = train_datagen.flow_from_directory(

        '../input/look4me/train/train',  # this is the target directory

        target_size=(64, 64),  # all images will be resized to 150x150

        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels



# this is a similar generator, for validation data

validation_generator = test_datagen.flow_from_directory(

        '../input/look4me/test/test',

        target_size=(64, 64),

        batch_size=batch_size)
def plot_res():

        # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(3))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()
history = model.fit_generator(

        train_generator,

        steps_per_epoch=2000 // batch_size,

        epochs=10,

        validation_data=validation_generator,

        validation_steps=800 // batch_size ,

        shuffle = True)
# Plot accuracy and loss over each epoch

plot_res()