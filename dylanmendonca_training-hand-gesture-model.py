# Importing import modules

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras import optimizers

import matplotlib.pyplot as plt

from skimage import io

print(os.listdir("../input/data/data"))
# Train, test and validation directories

train_dir = "../input/data/data/train"

val_dir = "../input/data/data/validation"

test_dir = "../input/data/data/test"
# Declaring variables

outputSize = len(os.listdir(train_dir)) # number of different gestures. Will determine number of units in final dense layer

epochs = 30 # Number of epochs
# Train Data Generator to do data augmentation on training images

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)
# Test Data Generator

test_datagen = ImageDataGenerator(rescale=1./255)
# Setting up the train generator to flow from the train directory

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size = (256,256),

    batch_size = 32,

    class_mode = 'categorical',

    color_mode = 'grayscale'

)



# Doing the same as above for the validation directory

val_generator = test_datagen.flow_from_directory(

    val_dir,

    target_size = (256,256),

    batch_size = 32,

    class_mode = 'categorical',

    color_mode = 'grayscale'

)
# Function to create keras model for different number of gestures

def create_model(outputSize):

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (256,256,1)))

    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))

    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))

    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))

    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = 2))

    model.add(Flatten())

    model.add(Dropout(rate = 0.5))

    model.add(Dense(512, activation = 'relu'))

    model.add(Dense(units = outputSize, activation = 'softmax'))

    model.compile(optimizer = optimizers.adam(lr=1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model
# Creating the model

model = create_model(outputSize)
# Summary of model

model.summary()
# Fitting the model to the data based on a 32 batch size

history = model.fit_generator(

    train_generator,

    steps_per_epoch=outputSize*1000/32,

    epochs=epochs,

    validation_data=val_generator,

    validation_steps=outputSize*500/32

)
# Plotting training acc/loss and val acc/loss

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

range_ep = epochs +1

epoch_x = range(1, range_ep)



plt.plot(epoch_x,acc,'bo',label="Training Acc")

plt.plot(epoch_x,val_acc,'b',label='Validation Acc')

plt.title('Training and Validation Acc')

plt.legend()

plt.figure()



plt.plot(epoch_x,loss,'bo',label="Training Loss")

plt.plot(epoch_x,val_loss,'b',label='Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()

plt.figure()



plt.show()
# Setting up the test generator to flow from the test directory

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size = (256,256),

    batch_size = 32,

    class_mode = 'categorical',

    color_mode = 'grayscale'

)
# Test accuracy and test loss calc

test_loss, test_acc = model.evaluate_generator(test_generator,steps = outputSize*500/32)

print("Test Acc:",test_acc)

print("Test Loss:",test_loss)
# Model weights and model

model.save_weights('my_model_weights.h5')

model.save("my_model.h5")