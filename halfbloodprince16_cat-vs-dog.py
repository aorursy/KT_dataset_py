import os
trainset = os.listdir("../input/dataset/dataset/training_set")
import numpy
import keras
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("../input/dataset/dataset/training_set",target_size = (64, 64),batch_size = 32,class_mode = 'binary')

test_set = test_datagen.flow_from_directory("../input/dataset/dataset/test_set",target_size = (64, 64),batch_size = 32,class_mode = 'binary')
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()
model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Flatten())

# Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit_generator(training_set,epochs = 10 ,validation_data = test_set,validation_steps = 2000)
model.save('weights.hdf5')
