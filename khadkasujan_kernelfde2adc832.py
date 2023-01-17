import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.preprocessing.image import ImageDataGenerator
model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(50, 50, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
model.add(Conv2D(64, (5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(44, activation='softmax'))
sgd = optimizers.SGD(lr=1e-2)
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '../input/dataset/dataset/train',
        target_size=(50, 50),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        '../input/dataset/dataset/test',
        target_size=(50, 50),
        batch_size=32,
        class_mode='categorical')
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
filepath="cnn_model_keras2.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint1]

model.fit_generator(
        train_generator,
        steps_per_epoch=value//32,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=value//32)

model.save('mymodel.h5')
