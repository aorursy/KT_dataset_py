# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.preprocessing import image

flower_classes = 5
# CNN structure
flower_recognizer_model = Sequential()
flower_recognizer_model.add(Conv2D(64,(3,3), input_shape = (64,64,3),strides=2,activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),strides=2,activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),activation = 'relu'))
flower_recognizer_model.add(Conv2D(64,(3,3),activation = 'relu'))
flower_recognizer_model.add(Flatten())
flower_recognizer_model.add(Dense(128,activation='relu'))
# Output layer
flower_recognizer_model.add(Dense(flower_classes,activation='softmax'))

flower_recognizer_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

flower_recognizer_model.summary()

# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input/flowers-recognition/flowers/flowers"))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory('../input/flowers-recognition/flowers/flowers',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

flower_recognizer_model.fit_generator(training_set, steps_per_epoch=100, epochs=10)

flower_recognizer_model.save('flower_recognition_model.h5')

flower_recognizer_model.load_weights('flower_recognition_model.h5')

image_to_recognize = image.load_img('../input/purple-rose-image/purple_rose.jpg', target_size=(64,64))
image_to_recognize = image.img_to_array(image_to_recognize)
image_to_recognize = np.expand_dims(image_to_recognize, axis = 0)
prediction = flower_recognizer_model.predict(image_to_recognize)

print(training_set.class_indices)
print(prediction)