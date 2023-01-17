# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



random.seed( 30 )



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#  genfromtxt is very slow

# from numpy import genfromtxt

# train_data = genfromtxt("/kaggle/input/digit-recognizer/train.csv", delimiter=',')

# test_data = genfromtxt("/kaggle/input/digit-recognizer/test.csv", delimiter=',')

# sample_submission = genfromtxt("/kaggle/input/digit-recognizer/sample_submission.csv", delimiter=',')

# print(train_data.shape, test_data.reshape([test_data.shape[0], -1]).shape, sample_submission.reshape([sample_submission.shape[0], -1]).shape)



train_data_from_file = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data_from_file = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sample_submission_from_file = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

print(train_data_from_file.shape, test_data_from_file.shape, sample_submission_from_file.shape)



train_label = np.array(train_data_from_file)[ : , : 1]

train_data = np.array(train_data_from_file)[ : , 1 : 785]

test_data = np.array(test_data_from_file)



print(train_data.shape, train_label.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train_data, train_label,

                                                    test_size=0.2,

                                                    random_state=0,

                                                    stratify=train_label)



print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def unflatten_image(number_of_images, length, width, channels, flattened_image):

    return flattened_image.reshape([number_of_images, length, width, channels])



x_train = unflatten_image(x_train.shape[0], 28 ,28, 1, x_train.flatten())

x_test = unflatten_image(x_test.shape[0], 28 ,28, 1, x_test.flatten())



print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
import numpy as np

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD



# Generate dummy data

y_train = keras.utils.to_categorical(y_train, num_classes=10)

y_test = keras.utils.to_categorical(y_test, num_classes=10)

def noramlize_4D_images(images):

    """

    :param images: numpy array of 4D images with 1st dimension as different inputs and last dimension as channels

    :return: 

    """

    images_min = images.min(axis=(1, 2), keepdims=True)

    images_max = images.max(axis=(1, 2), keepdims=True)

    return (images - images_min)/(images_max-images_min)



x_train = noramlize_4D_images(x_train)

x_test = noramlize_4D_images(x_test)
model = Sequential()

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.

# this applies 32 convolution filters of size 3x3 each.

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=64, epochs=10)

score = model.evaluate(x_test, y_test, batch_size=32)

print(score)
test_data_unflattened = unflatten_image(test_data.shape[0], 28 ,28, 1, test_data.flatten())
label = model.predict(test_data_unflattened, batch_size=64)
def get_labels_from_categorical(label):

    """

    :param label: 2D array with 1st dimension having input vector and 2nd dimension has one hot vectors

    :return: label with numeric value having shape [n, 1]

    """

    label =  np.argmax(label, axis=1)

    return label.reshape(label.shape[0],1)



label = get_labels_from_categorical(label)
def add_indices(input_array):

    """

    :param input_array: input array to which indices needs to be added

    :return: array with 1st column as indices starting from 1

    """

    indices = np.arange(1, input_array.shape[0]+1).reshape(input_array.shape[0], 1)

    return np.concatenate([indices, input_array], axis = 1)



label = add_indices(label)
np.savetxt('test.csv', label, delimiter=',', header= 'ImageId,Label', fmt='%1.i', comments='')