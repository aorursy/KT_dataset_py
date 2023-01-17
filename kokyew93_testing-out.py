import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import os

import random



%matplotlib inline
parasitized_data = os.listdir('../input/cell_images/cell_images/Parasitized/')

uninfected_data = os.listdir('../input/cell_images/cell_images/Uninfected/')
sample_uninfected = random.sample(uninfected_data,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('../input/cell_images/cell_images/Uninfected/'+sample_uninfected[i])

    ax[i//3,i%3].imshow(im)

    ax[i//3,i%3].axis('off')

f.suptitle('Uninfected')

plt.show()
sample_parasitized = random.sample(parasitized_data,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('../input/cell_images/cell_images/Parasitized/'+sample_parasitized[i])

    ax[i//3,i%3].imshow(im)

    ax[i//3,i%3].axis('off')

f.suptitle('Parasitized')

plt.show()
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,  

    zoom_range=0.2,        

    horizontal_flip=True,

    validation_split=0.3)  



train_generator = datagen.flow_from_directory(

    '../input/cell_images/cell_images',

    target_size = (150,150),

    batch_size=16,

    class_mode = 'binary',

    subset='training')



val_generator = datagen.flow_from_directory(

    '../input/cell_images/cell_images',

    target_size=(150,150),

    batch_size=16,

    class_mode='binary',

    subset='validation')

history = model.fit_generator(

    train_generator, 

    steps_per_epoch  = 100, 

    validation_data  = val_generator,

    validation_steps = 50,

    epochs = 10)
model.save('malaria.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

fig = plt.figure(figsize=(16,9))



plt.subplot(1, 2, 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()