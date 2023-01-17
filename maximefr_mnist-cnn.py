import numpy as np   # linear algebra

import pandas as pd  # data processing

import matplotlib.pyplot as plt  # figures

from keras.utils import to_categorical

%matplotlib inline
#sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")



train_data = train.iloc[:, 1:]                         # pixel data are indexed

train_labels = train['label'].values.astype('float32') # just a one-dimensional ndarray, with answers for 42000 samples



# return numpy representation of DataFrame object

test_data = test.to_numpy(dtype='float32') # does exactly the same as the line above but for the test dataset



# the last index is added for further use in augmentations

train_data = train_data.to_numpy(dtype='float32').reshape(train_data.shape[0], 28, 28, 1)

test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)



train_data = train_data/255  

test_data = test_data/255  



train_labels = to_categorical(train_labels)
from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



def define_model():



    model = Sequential()



    # Kernel size is the size of the filter matrix for our convolution.

    model.add(Conv2D(filters = 64, kernel_size = (5,5), padding ='Same', activation ='relu', input_shape = (28,28,1)))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters = 64, kernel_size = (5,5), padding ='Same', activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 32, kernel_size = (2,2), padding ='Same', activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters = 32, kernel_size = (2,2), padding ='Same', activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))



    # Flatten layer serves as a connection between the convolution and dense layers.

    model.add(Flatten())



    #Dense is a standard layer type

    model.add(Dense(128, activation = "relu"))

    model.add(Dropout(0.15))

    model.add(Dense(64, activation = "relu"))

    model.add(Dense(10, activation = "softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=0.01), metrics=['accuracy'])

    #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

    

    return model
fit = model.fit(train_data, train_labels, batch_size=32, epochs=30, verbose=1, validation_split=0.05, shuffle=True)
predictions = model.predict_classes(test_data, batch_size=32, verbose=0)
submissions = pd.DataFrame({'ImageId':list(range(1,len(predictions) + 1)), "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)