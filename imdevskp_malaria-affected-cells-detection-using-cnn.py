# file operations

import os

# to list files

import glob



# for numerical analysis

import numpy as np 

# to store and process in a dataframe

import pandas as pd 



# for ploting graphs

import matplotlib.pyplot as plt

# advancec ploting

import seaborn as sns



# image processing

import matplotlib.image as mpimg



# train test split

from sklearn.model_selection import train_test_split

# model performance metrics

from sklearn.metrics import confusion_matrix, classification_report



# utility functions

from tensorflow.keras.utils import to_categorical, plot_model

# process image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# sequential model

from tensorflow.keras.models import Sequential

# layers

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# callback functions

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# list of files in the dataset

os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images')
# list all the images in the directory Parasitized

parasitized = glob.glob('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/*.png')



# no. of files in the directory Parasitized

print('No. of files in the directory Parasitized', len(parasitized))



# first few images

parasitized[:5]
# list all the images in the directory Uninfected

uninfected = glob.glob('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/*.png')



# no. of files in the directory Uninfected

print('No. of files in the directory Uninfected', len(uninfected))



# first few images

uninfected[:5]
fig, ax = plt.subplots(figsize=(18, 8))

fig.suptitle('Parasitized cells', fontsize=24)



for ind, img_src in enumerate(parasitized[:30]):

    plt.subplot(3, 10, ind+1)

    img = plt.imread(img_src)

    plt.axis('off')

    plt.imshow(img)
fig, ax = plt.subplots(figsize=(18, 8))

fig.suptitle('Uninfected cells', fontsize=24)



for ind, img_src in enumerate(uninfected[:30]):

    plt.subplot(3, 10, ind+1)

    img = plt.imread(img_src)

    plt.axis('off')

    plt.imshow(img)
BATCH_SIZE = 100  # Number of training examples to process before updating our models variables

IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

TARGET_SIZE = 64

EPOCHS = 10
model = Sequential()



model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(2,2))



model.add(Flatten())



model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))



model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()
plt.figure(figsize=(5, 10))

plot_model(model, to_file="model.png")
datagen = ImageDataGenerator(rescale=1./255,

                             zoom_range=0.2,

                             horizontal_flip=True,

                             vertical_flip=True,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             validation_split=0.3)



train_data = datagen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',

                                         target_size=(IMG_SHAPE,IMG_SHAPE),

                                         batch_size=BATCH_SIZE,

                                         shuffle=True,

                                         class_mode='binary',

                                         subset='training')



validation_data = datagen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',

                                              target_size=(IMG_SHAPE,IMG_SHAPE),

                                              batch_size=BATCH_SIZE,

                                              shuffle=True,

                                              class_mode='binary',

                                              subset='validation')
# Instantiate an early stopping callback

early_stopping = EarlyStopping(monitor='val_loss', 

                               min_delta = 0.01,

                               patience=5)



# Instantiate a model checkpoint callback

model_save = ModelCheckpoint('best_model.hdf5',

                             monitor='val_loss',

                             mode='min',

                             save_best_only=True)
history = model.fit(train_data,

                    validation_data=validation_data,

                    epochs=EPOCHS,

                    verbose=1, 

                    callbacks=[early_stopping, model_save])
plt.figure(figsize=(14, 5))



plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')



plt.show()