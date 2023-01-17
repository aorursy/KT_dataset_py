import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from glob import glob



import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization

from tensorflow.keras.layers import ZeroPadding2D

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.callbacks import ReduceLROnPlateau



from sklearn.model_selection import train_test_split 



import keras

from keras.utils import np_utils

from keras.preprocessing import image

from keras.models import Model
import os

paths = os.listdir(path="../input")

print(paths)
data = "../input/chest-xray-pneumonia/chest_xray/train"
classes = ["NORMAL", "PNEUMONIA"]

train_data = glob(data+"/NORMAL/*.jpeg")

train_data += glob(data+"/PNEUMONIA/*.jpeg")
datagen = ImageDataGenerator(validation_split=0.2)

training_set = datagen.flow_from_directory(data, target_size = (226, 226), classes = classes, 

                                           class_mode = "categorical",subset='training')

validation_set = datagen.flow_from_directory(data, target_size = (226, 226), classes = classes, 

                                             class_mode = "categorical",subset='validation')
training_set.class_indices
input_shape=training_set.image_shape



model = Sequential()



model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dense(2, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
early_stopping = EarlyStopping('accuracy', patience=10, mode='max')

lr_plateau = ReduceLROnPlateau('accuracy', patience=10, verbose=2, mode='max', )

callbacks= [early_stopping,lr_plateau]
history = model.fit(training_set,epochs=30, shuffle=True, 

                    validation_data=validation_set, verbose=2,callbacks=callbacks)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()