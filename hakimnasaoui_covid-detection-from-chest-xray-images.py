from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D

from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense



from keras.models import Sequential, Model

#rom keras.applications.xception import Xception

#from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.optimizers import Adam, SGD, RMSprop



import tensorflow as tf



import os

import numpy as np

import pandas as np



import matplotlib.pyplot as plt

%matplotlib inline

DATASET_DIR = "../input/covid-19-x-ray-10000-images/dataset"



IMG_W = 150

IMG_H = 150

CHANNELS = 3



INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)

NB_CLASSES = 2

EPOCHS = 15

BATCH_SIZE = 8

os.listdir(DATASET_DIR)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])

#model.compile(Adam(lr=0.0001),loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

import glob

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



normal_images = []

for img_path in glob.glob(DATASET_DIR + '/normal/*'):

    normal_images.append(mpimg.imread(img_path))



fig = plt.figure()

fig.suptitle('normal')

plt.imshow(normal_images[0], cmap='gray') 

covid_images = []

for img_path in glob.glob(DATASET_DIR + '/covid/*'):

    covid_images.append(mpimg.imread(img_path))



fig = plt.figure()

fig.suptitle('covid')

plt.imshow(covid_images[0], cmap='gray') 

train_datagen = ImageDataGenerator(rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.4)



train_generator = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_H, IMG_W),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='training')



validation_generator = train_datagen.flow_from_directory(

    DATASET_DIR, # same directory as training data

    target_size=(IMG_H, IMG_W),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='validation')



history = model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples // BATCH_SIZE,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples // BATCH_SIZE,

    epochs = EPOCHS)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()