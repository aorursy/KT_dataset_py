### \author Ben Snow

### \version 1.0

### \date Last Revision 18/11/2019 \n



### \class ASE Autumn 2019

### \brief 

### @brief Simple autoencoder trained to recreate the mnist dataset

### Modified from :-

### Francois Chollet (14 May 2016). Building Autoencoders in Keras [online].

### [Accessed 2019]. Available from: "https://blog.keras.io/building-autoencoders-in-keras.html".





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import keras

from keras.layers import Input, Dense

from keras.models import Model

import tensorflow as tf
encoding_dim = 32 #number dimensions of encoded representation



input_img = Input(shape=(784,))  #input image dimensions



encoded = Dense(encoding_dim, activation='relu')(input_img) #encoded representation is stored here



decoded = Dense(784, activation='sigmoid')(encoded) #decoded representation is stored here



autoencoder = Model(input_img, decoded) #this is the autoencoder! It encodes then decodes the data to reproduce the input image
encoder = Model(input_img, encoded) #maps the input to the reduced representation
encoded_input = Input(shape=(encoding_dim,)) #declaring a variable for holding the reduced representation



decoded_layer = autoencoder.layers[-1] #isolating the last layer of the autoencoder (to be used as the last layer of the decoder)



decoder = Model(encoded_input, decoded_layer(encoded_input)) #creates the decoder model with dimensions defined by the size of the reduced representation
autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy') #configuring the optimiser and the loss function for the model

#adadelta is an adaptive gradient descent optimiser: https://arxiv.org/abs/1212.5701

#binary crossentropy loss function is used since the mnist dataset is being used

#mnist has 2 categories (black + white), hence 'binary': https://bit.ly/2PyiCst (derivation of loss fn)
x_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv") #reading in pre-split testing and training data from the mnist data set

x_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

x_test = x_test.drop(['label'], axis=1)

x_train = x_train.drop(['label'], axis=1)
x_train = x_train.astype('float32')/255. #normalising the training and testing data

x_test = x_test.astype('float32')/255.

#x_train.shape

x_train = pd.DataFrame(x_train).to_numpy()

x_test = pd.DataFrame(x_test).to_numpy()



x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1]))) #reshaping the data set to the dimensions of the autoencoder (784 nodes)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1])))

print (x_train.shape)

print (x_test.shape)
autoencoder.fit(x_train, x_train, epochs=5, batch_size=50, shuffle=True, validation_data=(x_test,x_test)) #training the autoencoder for 5 epochs splitting the data into shuffled batches of 50 images

encoded_imgs = encoder.predict(x_test)  #extracting the encoded and decoded images

decoded_imgs = decoder.predict(encoded_imgs)
import matplotlib.pyplot as plt



#comparing the first 10 input images vs. their corresponding decoded images after being fed through the trained network



n=10

plt.figure(figsize=(20,4))

for i in range(n):

    ax=plt.subplot(2, n, i+1)

    plt.imshow(x_test[i].reshape(28,28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    ax=plt.subplot(2, n, i+1+n)

    plt.imshow(decoded_imgs[i].reshape(28,28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()