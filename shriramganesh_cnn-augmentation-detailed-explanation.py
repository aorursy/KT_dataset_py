%matplotlib inline

%load_ext autoreload

%autoreload 2

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import tensorflow as tf

from math import pi
x_train = pd.read_csv("../input/csvTrainImages 13440x1024.csv" )

y_train = pd.read_csv("../input/csvTrainLabel 13440x1.csv")

x_test  = pd.read_csv("../input/csvTestImages 3360x1024.csv")

y_test  = pd.read_csv("../input/csvTestLabel 3360x1.csv")
x_train.info()
x_train = x_train.iloc[:,:].values

y_train = y_train.iloc[:,:].values

x_test = x_test.iloc[:,:].values

y_test = y_test.iloc[:,:].values
def visualize_images(df, img_size, number_of_images):

    plt.figure(figsize=(8,8))

    

    n_rows = df.shape[0]

    reshaped_df = df.reshape(df.shape[0], img_size, img_size)

    number_of_rows = number_of_images/5 if number_of_images%5 == 0 else (number_of_images/5) +1

    for i in range(number_of_images):

        plt.subplot(number_of_rows, 5, i+1, xticks=[], yticks=[])

        plt.imshow(reshaped_df[i], cmap='gray')

visualize_images(x_train, 32, 16)
def visualize_input(img, ax):

    img = img.reshape(32, 32)

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y],2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y]<thresh else 'black')



fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111, xticks=[], yticks=[])

visualize_input(x_train[0], ax)
# rescale [0,255] --> [0,1]

x_train = x_train.astype('float32')/255

x_test = x_test.astype('float32')/255

y_train = y_train.astype('int32')

y_test = y_test.astype('int32')
max_ = y_train.max()+1
import keras

from keras.utils import np_utils



# break training set into training and validation sets

(x_train, x_valid) = x_train[1000:], x_train[:1000]

(y_train, y_valid) = y_train[1000:], y_train[:1000]



# one-hot encode the labels

num_classes = len(np.unique(y_train))+1

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

y_valid = keras.utils.to_categorical(y_valid, num_classes)



# print shape of training set

print('x_train shape:', x_train.shape)



# print number of training, validation, and test images

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

print(x_valid.shape[0], 'validation samples')
x_train = x_train.reshape([-1, 32, 32, 1])

x_test = x_test.reshape([-1, 32, 32, 1])

x_valid = x_valid.reshape([-1, 32, 32, 1])
x_train.shape
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 

                        input_shape=x_train.shape[1:]))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(29, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 

                  metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



# create and configure augmented image generator

datagen_train = ImageDataGenerator(

    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)

    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)

    horizontal_flip=True) # randomly flip images horizontally



# fit augmented image generator on data

datagen_train.fit(x_train)
from keras.callbacks import ModelCheckpoint   



batch_size = 128

epochs = 100



# train the model

checkpointer = ModelCheckpoint(filepath='aug_model.weights.best.hdf5', verbose=1, 

                               save_best_only=True)

model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),

                    steps_per_epoch=x_train.shape[0] // batch_size,

                    epochs=epochs, verbose=2, callbacks=[checkpointer],

                    validation_data=(x_valid, y_valid),

                    validation_steps=x_valid.shape[0] // batch_size)
model.load_weights('aug_model.weights.best.hdf5')
score = model.evaluate(x_test, y_test, verbose=0)

print('\n', 'Test accuracy:', score[1]*100)