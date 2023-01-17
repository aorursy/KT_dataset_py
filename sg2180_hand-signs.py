import numpy as np

from PIL import Image

from matplotlib.pyplot import imshow

import os

import re



import keras.backend as backend

from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, ZeroPadding2D

from keras.models import Sequential

from keras.losses import categorical_crossentropy

from keras.activations import sigmoid, relu

from keras import utils

from keras.models import model_from_json





%matplotlib inline

BASE_PATH = "../input/fingers/fingers/"

TRAINING_DATA_PATH = BASE_PATH + "/train/"

TEST_DATA_PATH = BASE_PATH + "/test/"
training_files = os.listdir(TRAINING_DATA_PATH)

test_files = os.listdir(TEST_DATA_PATH)

print (len(training_files), len(test_files))
def fname_to_label(fname):

    return re.search("([0-5])(L|R).png", fname).group(1)
def load_data(path, fnames, max_files=1000):

    img_data = []

    img_labels = []

    for i, fname in enumerate(fnames):

        img = np.array(Image.open(path + fname), dtype='uint8')

        img_data.append(img)

        img_labels.append(int(fname_to_label(fname)))

        if i == max_files:

            break

    return np.array(img_data), np.array(img_labels)
X_train, y_train = load_data(TRAINING_DATA_PATH, training_files, len(training_files))

NUM_CLASSES = 6

X_test, y_test = load_data(TEST_DATA_PATH, test_files, len(test_files))

print (X_train.shape, X_test.shape)
print( y_train.shape, y_train[0:5])

print( y_test.shape, y_test[0:5])
index = 7

imshow(X_train[index])

print (y_train[index])
def preprocess_data(X, y):

    X = X/255

    if backend.image_data_format() == 'channels_first':

        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

        input_shape = (1, X.shape[1], X.shape[2])

    else:

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        input_shape = (X.shape[1], X.shape[2], 1)

        

    y = utils.to_categorical(y, NUM_CLASSES)

    

    return (X, y, input_shape)

    
X_train, y_train, input_shape = preprocess_data (X_train, y_train)

X_test, y_test, test_shape = preprocess_data (X_test, y_test)
print (X_train.shape, y_train.shape, input_shape)

print (X_test.shape, y_test.shape, test_shape)
# Build the model



def finger_model(input_shape):

    model = Sequential()

    model.add(ZeroPadding2D((3,3)))

    

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, name="conv0"))

    model.add(Activation('relu'))



    model.add(Conv2D(128, kernel_size=(7, 7), input_shape=input_shape, name="conv1"))

    model.add(Activation('relu'))



    #model.add(BatchNormalization(axis=3, name='bn1'))

    model.add(MaxPool2D((4,4), name='maxpool1'))

    

    model.add(Flatten())

    

    #model.add(Dense(128, activation='sigmoid', name='fc1'))



    model.add(Dense(NUM_CLASSES, activation='softmax', name='fc2'))

    

    model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model

              
f_model = finger_model(input_shape)
f_model.fit(X_train, y_train, batch_size=50, epochs=3, verbose=1)

print (f_model.evaluate(X_test, y_test))