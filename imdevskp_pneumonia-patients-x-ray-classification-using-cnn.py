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

from tensorflow.keras.utils import to_categorical

# process image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# sequential model

from tensorflow.keras.models import Sequential

# layers

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# current working directory

os.getcwd()
# no. of files



def list_files(startpath):

    for root, dirs, files in os.walk(startpath):

        level = root.replace(startpath, '').count(os.sep)

        indent = ' ' * 4 * (level)

        print('{}{}'.format(indent, os.path.basename(root)), '-', len(os.listdir(root)))

        

folder = '../input/chest-xray-pneumonia/chest_xray/chest_xray'

list_files(folder)
# list of files in the dataset

os.listdir('../input/chest-xray-pneumonia/chest_xray/chest_xray')
# path to each directory



base_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray'

train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')

test_dir = os.path.join(base_dir, 'test')
# list of files in the train dataset

os.listdir(train_dir)
# path to both folder in the train directory

p_dir = os.path.join(train_dir, 'PNEUMONIA')



# list of pnemonia affected and normal images

p_imgs = os.listdir(p_dir)



# print no. of images

len(p_imgs)
# path to both folder in the train directory

n_dir = os.path.join(train_dir, 'NORMAL')



# list of pnemonia affected and normal images

n_imgs = os.listdir(n_dir)



# print no. of images

len(n_imgs)
# show pnemonia affected lungs X-rays



fig, ax = plt.subplots(figsize=(18, 6))

fig.suptitle('Pnemonia affected lungs', fontsize=24)



for i, img_path in enumerate(p_imgs[:24]):

    plt.subplot(3, 8, i+1)

    img = mpimg.imread(os.path.join(p_dir, img_path))

    plt.axis('off')

    plt.imshow(img)

plt.show()
# show normal lungs X-rays



fig, ax = plt.subplots(figsize=(18, 6))

fig.suptitle('Normal lungs', fontsize=24)



for i, img_path in enumerate(n_imgs[:24]):

    plt.subplot(3, 8, i+1)

    img = mpimg.imread(os.path.join(n_dir, img_path))

    plt.axis('off')

    plt.imshow(img)

plt.show()
train_data_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train'

validation_data_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val'

test_data_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test'



IMG_SIZE = 64

BATCH_SIZE = 32

TARGET_SIZE = 64

EPOCHS = 10
model = Sequential()



model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(2,2))



model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()
datagen = ImageDataGenerator(rescale=1./255,

                             zoom_range=0.2,

                             horizontal_flip=True,

                             vertical_flip=True,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             validation_split=0.3)



test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_data_dir,

                                              target_size=(IMG_SIZE, IMG_SIZE),

                                              batch_size=BATCH_SIZE,

                                              shuffle=True,

                                              class_mode='binary', 

                                              subset='training')



validation_generator = datagen.flow_from_directory(train_data_dir,

                                                   target_size=(IMG_SIZE, IMG_SIZE),

                                                   batch_size=BATCH_SIZE,

                                                   shuffle=True,

                                                   class_mode='binary', 

                                                   subset='validation')



test_generator = test_datagen.flow_from_directory(test_data_dir,

                                                  target_size=(IMG_SIZE, IMG_SIZE),

                                                  batch_size=BATCH_SIZE,

                                                  shuffle=True,

                                                  class_mode='binary')
history = model.fit_generator(train_generator,

                              validation_data=validation_generator,

                              epochs=EPOCHS,

                              verbose=1)
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
# evaluate model

model.evaluate(test_generator)