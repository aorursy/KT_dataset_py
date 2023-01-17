# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import os

from tensorflow.keras.preprocessing import image

from zipfile import ZipFile 
# importing libraries for Deep Learning

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_dir="../input/dogs-cats-images/dog vs cat/dataset/test_set"

train_dir="../input/dogs-cats-images/dog vs cat/dataset/training_set"



train_dir_cats = train_dir + '/cats'

train_dir_dogs = train_dir + '/dogs'

test_dir_cats = test_dir + '/cats'

test_dir_dogs = test_dir + '/dogs'
print('number of cats training images - ',len(os.listdir(train_dir_cats)))

print('number of dogs training images - ',len(os.listdir(train_dir_dogs)))

print('number of cats testing images - ',len(os.listdir(test_dir_cats)))

print('number of dogs testing images - ',len(os.listdir(test_dir_dogs)))
data_generator = ImageDataGenerator(rescale = 1.0/255.0, zoom_range = 0.2)
batch_size = 32

training_data = data_generator.flow_from_directory(directory = train_dir,

                                                   target_size = (64, 64),

                                                   batch_size = batch_size,

                                                   class_mode = 'binary')

testing_data = data_generator.flow_from_directory(directory = test_dir,

                                                  target_size = (64, 64),

                                                  batch_size = batch_size,

                                                  class_mode = 'binary')
# preparing the layers in the Convolutional Deep Neural Network

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training_data.image_shape))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.3))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 126, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.15))

model.add(Flatten())

model.add(Dense(units = 32, activation = 'relu'))

model.add(Dropout(rate = 0.15))

model.add(Dense(units = 64, activation = 'relu'))

model.add(Dropout(rate = 0.1))

model.add(Dense(units = len(set(training_data.classes)), activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
fitted_model = model.fit_generator(training_data,

                        steps_per_epoch = 1000,

                        epochs = 25,

                        validation_data = testing_data,

                        validation_steps = 1000)
# plotting accuracy and validation accuracy

accuracy = fitted_model.history['acc']

plt.plot(range(len(accuracy)), accuracy, 'bo', label = 'accuracy')

plt.legend()
# testing the model

def testing_image(image_directory):

    test_image = image.load_img(image_directory, target_size = (64, 64))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(x = test_image)

    print(result)

    if result[0][0]  >= 0.5:

        prediction = 'Dog'

    else:

        prediction = 'Cat'

    return prediction
print(testing_image(test_dir + '/dogs/dog.5000.jpg'))