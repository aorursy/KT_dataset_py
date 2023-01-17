# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

from keras.layers import Dropout
# fix random seed for reproducibility

seed = 12

np.random.seed(seed)



# Import Dataset, split and reshape

fashion_mnist_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

fashion_mnist_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")



X_train = fashion_mnist_train.iloc[:,1:].values

y_train = fashion_mnist_train.label.values



X_test = fashion_mnist_test.iloc[:,1:].values

y_test = fashion_mnist_test.label.values



# flatten 28*28 images to a 784 vector for each image

num_pixels = X_train.shape[1] 

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')

X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255
# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
# define baseline model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# build the model

model = baseline_model()

# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=200, verbose=2)

# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))