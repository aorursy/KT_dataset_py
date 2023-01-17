from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

import os

import cv2

import numpy as np
# Defining initial variables

parasitized_input_dir = os.listdir('../input/cell_images/cell_images/Parasitized') 

uninfected_input_dir = os.listdir('../input/cell_images/cell_images/Uninfected')

img_height = 50

img_width = 50

batch_size = 32

no_of_epochs = 20
image_data = [] 

labels = []

# for reading images of parasitized cells

for p in parasitized_input_dir:

    try:

        img = cv2.imread('../input/cell_images/cell_images/Parasitized/' + p)

        img = cv2.resize(img, (img_width, img_height)) # all images will be resized to 50*50

        image_data.append(img)

        labels.append(1)

    except:

        print("Error!!") # if the input data is not in desired format it will generate an error
# for reading images of Uninfected cells

for u in uninfected_input_dir:

    try:

        img = cv2.imread('../input/cell_images/cell_images/Uninfected/' + u)

        img = cv2.resize(img, (img_width, img_height))

        image_data.append(img)

        labels.append(0)

    except:

        print("Error!!")
image_data = np.array(image_data)

labels = np.array(labels)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(image_data, labels, test_size=0.2, random_state=101)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, shear_range=0.2, zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen.fit(X_train)

test_datagen.fit(X_test)
# Defining the NN Architecture

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(img_height, img_width, 3)))

model.add(Conv2D(16, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))
# Compiling the model

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
model.fit_generator(train_datagen.flow(X_train, Y_train), 

                    steps_per_epoch=len(X_train)//batch_size,

                    epochs=no_of_epochs,

                    validation_data=test_datagen.flow(X_test, Y_test),

                    validation_steps=len(X_test)//batch_size

                   )