import os
import cv2
import pandas as pd
import numpy as np
from scipy import ndimage
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
train_dir = Path('../input/10-monkey-species/training/training/')
test_dir = Path('../input/10-monkey-species/validation/validation/')
#label info
cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv("../input/10-monkey-species/monkey_labels.txt", names=cols, skiprows=1)
labels
labels.info()
from PIL import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Training generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(150,150),
                                                    batch_size= 64,
                                                    seed=1,
                                                    shuffle=True,
                                                    class_mode='categorical')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(150,150), 
                                                  batch_size=64,
                                                  seed=1,
                                                  shuffle=False,
                                                  class_mode='categorical')

train_num = train_generator.samples
validation_num = validation_generator.samples
num_classes = 10

monkey_model = Sequential()
monkey_model.add(Conv2D(32,(3,3), input_shape=(150,150,3), activation='relu'))
monkey_model.add(MaxPooling2D(pool_size=(2,2)))

monkey_model.add(Conv2D(32,(3,3), activation='relu'))
monkey_model.add(MaxPooling2D(pool_size=(2,2)))

monkey_model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
monkey_model.add(Conv2D(64,(3,3), activation='relu'))
monkey_model.add(MaxPooling2D(pool_size=(2,2)))
monkey_model.add(Dropout(0.25))

monkey_model.add(Flatten())
monkey_model.add(Dense(512))
monkey_model.add(Activation('relu'))
monkey_model.add(Dropout(0.5))
monkey_model.add(Dense(num_classes))
monkey_model.add(Activation('softmax'))
monkey_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['acc'])
monkey_model.summary()
# gotta figure out what this means...

filepath=str(os.getcwd()+"/model.h5f")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# = EarlyStopping(monitor='val_acc', patience=15)
callbacks_list = [checkpoint]#, stopper]
batch_size = 64

monkey_generator = monkey_model.fit_generator(train_generator,
                                             steps_per_epoch = train_num // batch_size,
                                              epochs = 100,
                                              validation_data = train_generator,
                                              validation_steps = validation_num // batch_size,
                                              callbacks = callbacks_list,
                                              verbose = 1)
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.plot(monkey_generator.history['acc'])
plt.plot(monkey_generator.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(monkey_generator.history['loss'])
plt.plot(monkey_generator.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()