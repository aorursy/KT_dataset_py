import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
import random
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
TRAINING_DIR = '/kaggle/input/dogs-cats-images/dataset/training_set/'
num_cat_images_train = len(os.listdir(os.path.join(TRAINING_DIR,'cats')))
num_dog_images_train = len(os.listdir(os.path.join(TRAINING_DIR,'dogs')))
print('Cat images:',num_cat_images_train)
print('Dog images:',num_dog_images_train)
print('Total:',num_cat_images_train + num_dog_images_train)
TESTING_DIR = '/kaggle/input/dogs-cats-images/dataset/test_set/'
num_cat_images_test = len(os.listdir(os.path.join(TESTING_DIR,'cats')))
num_dog_images_test = len(os.listdir(os.path.join(TESTING_DIR,'dogs')))
print('Cat images:',num_cat_images_test)
print('Dog images:',num_dog_images_test)
print('Total:',num_cat_images_test+num_dog_images_test)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size = 128,
    target_size = (150,150),
    class_mode = 'binary'
)
validation_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_generator = validation_datagen.flow_from_directory(
    TESTING_DIR,
    batch_size = 128,
    target_size = (150,150),
    class_mode = 'binary'
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Conv2D,Dropout,MaxPooling2D
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3),activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3),activation='relu'),
    MaxPooling2D(2, 2),
    GlobalAveragePooling2D(),
    Dense(512,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
early_cb = EarlyStopping(monitor='val_loss',patience=7)
rlrp_cb = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.5,min_lr=0.000001)
model_cp = ModelCheckpoint(filepath='/kaggle/working/my_model.h5',monitor='val_acc',save_best_only=True,save_weights_only=True,mode='max')
history = model.fit_generator(train_generator,
                              epochs=20,
                              verbose=1,
                              validation_data=validation_generator,
                             callbacks=[early_cb,rlrp_cb,model_cp])
