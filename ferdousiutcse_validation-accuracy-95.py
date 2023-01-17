# This is a personal project to create a simple convolution neural network

# The hyperparamters selected can be improved through more rigorous analysis

# I kept the model simple yet achieved a very good accuracy on the test dataset

# the dataset was split into two parts: first 80% data for training, last 20% data for testing

# V4 Commit: model checkpoint added
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import cv2

import os

import shelve

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



storage = shelve.open('saved_data')



print(os.listdir("../input/"))



parasitized_dir = '../input/cell_images/cell_images/Parasitized/'

uninfected_dir = '../input/cell_images/cell_images/Uninfected/'



parasitized_data = []

parasitized_label = []



for x in tqdm(os.listdir(parasitized_dir)):

    try:

        img = cv2.resize(cv2.imread(parasitized_dir + str(x), cv2.IMREAD_GRAYSCALE), (64, 64))

        parasitized_data.append(img)

        parasitized_label.append('parasitized')

    except:

        continue



storage['parasitized_data'] = parasitized_data

storage['parasitized_label'] = parasitized_label





uninfected_data = []

uninfected_label = []



for x in tqdm(os.listdir(uninfected_dir)):

    try:

        img = cv2.resize(cv2.imread(uninfected_dir + str(x), cv2.IMREAD_GRAYSCALE), (64, 64))

        uninfected_data.append(img)

        uninfected_label.append('uninfected')

    except:

        continue



storage['uninfected_data'] = uninfected_data

storage['uninfected_label'] = uninfected_label

def label_img(label):

    img = []

    for x in label:

        if x == 'parasitized': img.append([1, 0])

        else: img.append([0, 1])



    return np.array(img)





data = storage['parasitized_data'] + storage['uninfected_data']

label = storage['parasitized_label'] + storage['uninfected_label']

label = label_img(label)



training_data = []

for i in tqdm(range(0, np.shape(data)[0])):

    training_data.append([np.array(data[i]), label[i]])



random.shuffle(training_data)



n_samples = np.shape(training_data)[0]

n_train = int(n_samples * 0.8)

n_test = int(n_samples * 0.2)

n_valid = int(n_samples * 0.2)



train_x = []

train_y = []

test_x = []

test_y = []



for x, y in training_data[0:n_train]:

    train_x.append(x)

    train_y.append([y[0], y[1]])



for x, y in training_data[n_train:(n_train + n_test)]:

    test_x.append(x)

    test_y.append(y)





train_x = np.reshape(train_x, newshape=(np.shape(train_x)[0], 64, 64, 1)) / 255

test_x = np.reshape(test_x, newshape=(np.shape(test_x)[0], 64, 64, 1)) / 255

train_y = np.array(train_y)

test_y = np.array(test_y)
from keras.models import Sequential, load_model

from keras.optimizers import nadam, adadelta

from keras.layers import Convolution2D, BatchNormalization, Dropout, Dense, Flatten, MaxPool2D

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D

from keras.callbacks import CSVLogger, ModelCheckpoint





# storage = shelve.open('saved_data')

csv_logger = CSVLogger('log.csv', append=True, separator=';')



filepath = "best_model.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callback_list = [csv_logger, checkpoint]





def create_model():

    model = Sequential()

    model.add(Conv2D(64, 3, activation='relu', strides=1, use_bias=True, input_shape=(64, 64, 1)))

    model.add(MaxPool2D())

    model.add(BatchNormalization())



    model.add(Conv2D(64, 3, activation='relu', strides=1, use_bias=True))

    model.add(MaxPool2D())

    model.add(BatchNormalization())



    model.add(Conv2D(64, 3, activation='relu', strides=1, use_bias=True))

    model.add(MaxPool2D())

    model.add(BatchNormalization())



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.6))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.1))



    model.add(Dense(2, activation='softmax'))

    return model



model = create_model()

# model = load_model('partly_trained')



ada = adadelta(lr=1, rho=0.90, epsilon=None, decay=0.001)

model.compile(optimizer=ada, loss='binary_crossentropy', metrics=['acc'])

model.fit(train_x, train_y, batch_size=32, epochs=60, callbacks=callback_list, validation_data=(test_x, test_y))

# model.save('partly_trained_1')



# model.load_weights("best_model.hdf5")

# print(model.evaluate(test_x, test_y))
# model = load_model('partly_trained')

model.load_weights("best_model.hdf5")

print(model.evaluate(test_x, test_y))
print(model.summary())

# Display the predicted labels

import matplotlib.pyplot as plt

# model = load_model('partly_trained')



predicted = model.predict(test_x)

predicted_arg = np.argmax(predicted, axis=1)



fig = plt.figure(figsize = (10, 10))



count = 1

for i in range(16):

    z = fig.add_subplot(4, 4, count)

    count += 1



    org = np.reshape(test_x[i], newshape=(64, 64))

    z.imshow(org * 255, cmap='gray')

    z.axes.get_xaxis().set_visible(False)

    z.axes.get_yaxis().set_visible(False)

    if predicted_arg[i] == 0:

        plt.title('Parasitized')

    else:

        plt.title('Uninfected')



plt.show()