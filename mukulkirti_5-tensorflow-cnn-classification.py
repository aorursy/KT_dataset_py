import numpy as np 

import mnist #Get data set from 

import matplotlib.pyplot as plt #Graph

from keras.utils import to_categorical

from sklearn.externals import joblib

import tensorflow.compat.v1 as tf

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import random as rn





train_images = mnist.train_images() #training data images

train_labels = mnist.train_labels() #training data labels

test_images = mnist.test_images() #training data images

test_labels = mnist.test_labels() #training data labels

x_train=train_images.reshape(60000, 28, 28,1)

train_labels=train_labels.reshape(60000,1)

x_test=test_images.reshape(10000, 28, 28,1)

test_labels=test_labels.reshape(10000, 1)
print("Training set (images) shape: {shape}".format(shape=train_images.shape))

print("Training set (labels) shape: {shape}".format(shape=train_labels.shape))



# Shapes of test set

print("Test set (images) shape: {shape}".format(shape=test_images.shape))

print("Test set (labels) shape: {shape}".format(shape=test_labels.shape))
from keras.utils import to_categorical

y_train= to_categorical(train_labels)

y_test= to_categorical(test_labels)

y_train


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(10, activation = "softmax"))
from keras.callbacks import ReduceLROnPlateau

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
batch_size=15

epochs=15
# Befor tain Load the TensorBoard notebook extension for visualization

%load_ext tensorboard

# Clear any logs from previous runs

!rm -rf ./train/ 

#import datetime



log_dir = "train"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['accuracy'])

plt.plot(History.history['val_accuracy'])

plt.title('Training and Test Accuracy')

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.legend()

plt.figure()

plt.show()