import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
np.random.seed(2020)
!mkdir train test

!unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip -d train

!unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
import random

import json

import csv



from matplotlib import pyplot as plt

%matplotlib inline



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.preprocessing import image

from keras import optimizers
TRAIN_DIR = './train/train/'

TEST_DIR = './test/test'



ROWS = 150

COLS = 150

CHANNELS = 3



BATCH_SIZE=64



# HyperParams

EPOCHS=5

train_steps = len(os.listdir(TRAIN_DIR))/BATCH_SIZE

validation_steps = len(os.listdir(TEST_DIR))/BATCH_SIZE

lr=1e-4
original_train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]



# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

original_train_images = train_dogs[:12000] + train_cats[:12000]

# test_images =  test_images[:100]



# section = int(len(original_train_images) * 0.8)

train_images = original_train_images[:18000]

validation_images = original_train_images[18000:]
len(train_images)
def plot_arr(arr):

    plt.figure()

    plt.imshow(image.array_to_img(arr))

    plt.show()



def plot(img):

    plt.figure()

    plt.imshow(img)

    plt.show()

    

def prep_data(images):

    count = len(images)

    X = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.float32)

    y = np.zeros((count,), dtype=np.float32)

    

    for i, image_file in enumerate(images):

        img = image.load_img(image_file, target_size=(ROWS, COLS))

        X[i] = image.img_to_array(img)

        if 'dog' in image_file:

            y[i] = 1.

        if i%1000 == 0: print('Processed {} of {}'.format(i, count))

    

    return X, y
X_train, y_train = prep_data(train_images)
X_validation, y_validation = prep_data(validation_images)
train_datagen = image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



validation_datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(

    X_train,

    y_train,

    batch_size=BATCH_SIZE)



validation_generator = validation_datagen.flow(

    X_validation,

    y_validation,

    batch_size=BATCH_SIZE)
def create_custom_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(ROWS, COLS, CHANNELS)))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    return model
model = create_custom_model()

model.summary()
model.compile(loss='binary_crossentropy',

             optimizer=optimizers.Adam(lr=lr),

             metrics=['accuracy'])
history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=EPOCHS,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color='red')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model = keras.models.Sequential()

model.add(keras.applications.VGG16(include_top=False, pooling='max', weights='imagenet'))

model.add(Dense(1, activation='sigmoid'))

# ResNet-50 model is already trained, should not be trained

model.layers[0].trainable = True



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.Adam(lr=lr),

             metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=EPOCHS,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation acc', color='red')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model = keras.models.Sequential()

model.add(keras.applications.VGG19(include_top=False, pooling='max', weights='imagenet'))

model.add(Dense(1, activation='sigmoid'))

# ResNet-50 model is already trained, should not be trained

model.layers[0].trainable = True



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.Adam(lr=lr),

             metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=EPOCHS,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color='red')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model = keras.models.Sequential()

model.add(keras.applications.ResNet50(include_top=False, pooling='max', weights='imagenet'))

model.add(Dense(1, activation='sigmoid'))

# ResNet-50 model is already trained, should not be trained

model.layers[0].trainable = True



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.Adam(lr=lr),

             metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=EPOCHS,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc', color='red')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model = keras.models.Sequential()

model.add(keras.applications.ResNet101(include_top=False, pooling='max', weights='imagenet'))

model.add(Dense(1, activation='sigmoid'))

# ResNet-50 model is already trained, should not be trained

model.layers[0].trainable = True



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.Adam(lr=lr),

             metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=EPOCHS,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color='red')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')

plt.title('Training and validation loss')

plt.legend()

plt.show()
!rm -rf train

!rm -rf test