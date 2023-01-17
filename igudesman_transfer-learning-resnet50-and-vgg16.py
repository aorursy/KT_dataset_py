import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import math
BASE_PATH = '/kaggle/input/10-monkey-species/'
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
import glob
import random
import matplotlib.pyplot as plt
    

def get_image(dataset, class_num):
    reg_expression = BASE_PATH + '{0}/{0}/n{1}/*.jpg'.format(dataset, class_num)
    all_paths = glob.glob(reg_expression)
    img_path = all_paths[random.randint(0, len(all_paths) - 1)]
    print('Image Path: {0}'.format(img_path))
    img = load_img(img_path)
    img.show()
    img = img_to_array(img)
    print('Image Shape: {0}'.format(img.shape))
    plt.imshow(img/255.0)
    
get_image('training', 0)
datagen_with_aug = ImageDataGenerator(horizontal_flip=True,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      zoom_range=0.2)

datagen_with_no_aug = ImageDataGenerator()

train = datagen_with_aug.flow_from_directory(directory='/kaggle/input/10-monkey-species/training/training/', 
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE), 
                                               color_mode='rgb',
                                               class_mode='categorical',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

validation = datagen_with_no_aug.flow_from_directory(directory='/kaggle/input/10-monkey-species/validation/validation/',
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE), 
                                                    color_mode='rgb',
                                                    class_mode='categorical', 
                                                    batch_size=1)
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop
# DICH

model = Sequential()

model.add(Conv2D(10, (10, 10), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(20, (5, 5), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(generator=train, 
                    steps_per_epoch=math.ceil(train.samples / BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=validation)
# DICH

from tensorflow.keras.applications import ResNet50

transfer_model = Sequential()
transfer_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
transfer_model.add(Dense(10, activation='softmax'))

transfer_model.layers[0].trainable = False

transfer_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

transfer_model.fit_generator(generator=train, 
                    steps_per_epoch=math.ceil(train.samples / BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=validation)
# Accuracy: ~0.9

other_model = Sequential()
other_model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation='relu'))
other_model.add(MaxPooling2D(pool_size=(2, 2)))

other_model.add(Conv2D(32, (3, 3), activation='relu'))
other_model.add(MaxPooling2D(pool_size=(2, 2)))

other_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
other_model.add(Conv2D(64, (3, 3), activation='relu'))
other_model.add(Conv2D(128, (5, 5), activation='relu'))
other_model.add(MaxPooling2D(pool_size=(2, 2)))
other_model.add(Dropout(0.25))

other_model.add(Flatten())
other_model.add(Dense(512, activation='relu'))
other_model.add(Dropout(0.5))
other_model.add(Dense(512, activation='relu'))
other_model.add(Dropout(0.5))
other_model.add(Dense(10, activation='softmax'))

other_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

other_model.fit_generator(generator=train, 
                    steps_per_epoch=math.ceil(train.samples / BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=validation)
# Accuracy: 0.9338

from tensorflow.keras.applications import VGG16

base_model = VGG16(input_shape=(224, 224, 3), weights='imagenet')

base_model_output = base_model.layers[-2].output

# add new layers 
x = Dropout(0.5)(base_model_output)
output = Dense(10, activation='softmax')(x)

base_model = Model(base_model.input, output)

for layer in base_model.layers[:-1]:
    layer.trainable=False
    
optimizer = RMSprop(0.001)
base_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


base_model.fit_generator(generator=train, 
                    steps_per_epoch=math.ceil(train.samples / BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=validation)
