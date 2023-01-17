import numpy as np

np.random.seed(1000)

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split
import keras

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

from keras.models import Sequential
import os

import cv2

from PIL import Image
DATA_DIR = '../input/cell_images/cell_images/'

SIZE = 64

dataset = []

label = []
parasitized_images = os.listdir(DATA_DIR + 'Parasitized/')

for i, image_name in enumerate(parasitized_images):

    try:

        if (image_name.split('.')[1] == 'png'):

            image = cv2.imread(DATA_DIR + 'Parasitized/' + image_name)

            image = Image.fromarray(image, 'RGB')

            image = image.resize((SIZE, SIZE))

            dataset.append(np.array(image))

            label.append(0)

    except Exception:

        print("Could not read image {} with name {}".format(i, image_name))
uninfected_images = os.listdir(DATA_DIR + 'Uninfected/')

for i, image_name in enumerate(uninfected_images):

    try:

        if (image_name.split('.')[1] == 'png'):

            image = cv2.imread(DATA_DIR + 'Uninfected/' + image_name)

            image = Image.fromarray(image, 'RGB')

            image = image.resize((SIZE, SIZE))

            dataset.append(np.array(image))

            label.append(1)

    except Exception:

        print("Could not read image {} with name {}".format(i, image_name))
plt.figure(figsize = (20, 12))

for index, image_index in enumerate(np.random.randint(len(parasitized_images), size = 5)):

    plt.subplot(1, 5, index+1)

    

    plt.imshow(dataset[image_index])
plt.figure(figsize = (20, 12))

for index, image_index in enumerate(np.random.randint(len(uninfected_images), size = 5)):

    plt.subplot(1, 5, index+1)

    plt.imshow(datasetlen(parasitized_images))
classifier = None

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))

classifier.add(BatchNormalization(axis = -1))

classifier.add(Dropout(0.2))

classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))

classifier.add(BatchNormalization(axis = -1))

classifier.add(Dropout(0.2))

classifier.add(Flatten())

classifier.add(Dense(activation = 'relu', units=512))

classifier.add(BatchNormalization(axis = -1))

classifier.add(Dropout(0.2))

classifier.add(Dense(activation = 'relu', units=256))

classifier.add(BatchNormalization(axis = -1))

classifier.add(Dropout(0.2))

classifier.add(Dense(activation = 'sigmoid', units=2))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(classifier.summary())
from keras.utils import to_categorical



X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 0)
history = classifier.fit(np.array(X_train), 

                         y_train, 

                         batch_size = 64, 

                         verbose = 2, 

                         epochs = 50, 

                         validation_split = 0.1,

                         shuffle = False)
print("Test_Accuracy: {:.2f}%".format(classifier.evaluate(np.array(X_test), np.array(y_test))[1]*100))
from keras.preprocessing.image import ImageDataGenerator



train_generator = ImageDataGenerator(rescale = 1/255,

                                     zoom_range = 0.3,

                                     horizontal_flip = True,

                                     rotation_range = 30)



test_generator = ImageDataGenerator(rescale = 1/255)



train_generator = train_generator.flow(np.array(X_train),

                                       y_train,

                                       batch_size = 64,

                                       shuffle = False)



test_generator = test_generator.flow(np.array(X_test),

                                     y_test,

                                     batch_size = 64,

                                     shuffle = False)
history = classifier.fit_generator(train_generator,

                                   steps_per_epoch = len(X_train)/64,

                                   epochs = 50,

                                   shuffle = False)
print("Test_Accuracy(after augmentation): {:.2f}%".format(classifier.evaluate_generator(test_generator, steps = len(X_test), verbose = 1)[1]*100))