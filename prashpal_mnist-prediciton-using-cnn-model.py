# Import required packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import tensorflow as tf

from keras.models import Model

from keras.layers import Dense, Input

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten

from keras import backend as k

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load train data

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
# Divide into test and dev set

# Load test data

test = pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
# Convert train and test data into (num_images, img_rows, img_cols) format

x_train = (train.iloc[:,1:].values).astype('float32')

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = test.values.astype('float32')

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)



# Convert test labels into (num_images, labels) format

y_train = (train.iloc[:,0].values).astype('int32')



# Visualize data

for i in range(6, 9):

    plt.subplot(4, 4, i-5)

    plt.imshow(np.squeeze(x_train[i+101]), cmap=plt.get_cmap('gray'))

    plt.title(y_train[i+101])
# Normalize the train and test data

x_train = x_train/255

x_test = x_test/255



# Visualize again

for i in range(6, 9):

    plt.subplot(4, 4, i-5)

    plt.imshow(np.squeeze(x_train[i+101]), cmap=plt.get_cmap('gray'))

    plt.title(y_train[i+101])
# One hot encoding

y_train = keras.utils.to_categorical(y_train)
# Input shape

imgrows = x_train.shape[1]

imgcols = x_train.shape[2]

input_shape = (imgrows, imgcols, 1)

print(input_shape)
# CNN model

input_img = Input(shape=input_shape)

layer1 = Conv2D(32, (3,3), activation='relu') (input_img)

layer2 = Conv2D(64, (3,3), activation='relu') (layer1)

layer3 = MaxPooling2D(pool_size=(3,3)) (layer2)

layer4 = Dropout(0.5) (layer3)

layer5 = Flatten() (layer4)

layer6 = Dense(250, activation='relu') (layer5)

layer7 = Dense(10, activation='softmax') (layer6)
# Compile and fit model

cnn_model = Model(inputs=input_img, outputs=layer7)

cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = cnn_model.fit(x_train, y_train, epochs=70, batch_size=500)

plt.plot(history.history['loss'])
# predict results

results = cnn_model.predict(x_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,(x_test.shape[0]+1)),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_prediction.csv",index=False)