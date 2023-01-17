import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import keras 

from keras.datasets import mnist

from keras.models import Model

from keras.layers import Input,Dense
def load_images():

    (x_train,_),(x_test,_) = mnist.load_data()

    return x_train,x_test
def normalize(x_train,x_test):

    x_train = keras.utils.normalize(x_train)

    x_test = keras.utils.normalize(x_test)

    return x_train,x_test
def get_shape(x_train,x_test):

    return x_train.shape,x_test.shape
def reshape_data(x_train,x_test):

    x_train_re = x_train.reshape(len(x_train),np.prod(x_train.shape[1:]))

    x_test_re = x_test.reshape(len(x_test),np.prod(x_test.shape[1:]))

    return x_train_re,x_test_re
def encoder(x_train_re,x_test_re,epochs=80):

    ## image column is row in the input

    ## image row will be set to infinity so we can push any ammount of value

    ##symmetrical layer start here

    input_img = Input(shape=(784,))

    encoded = Dense(units=128,activation='relu')(input_img)

    encoded = Dense(units=64,activation='relu')(encoded)

    encoded = Dense(units=32,activation='relu')(encoded) ## this is the center layer

    decoded = Dense(units=64,activation='relu')(encoded)

    decoded = Dense(units=128,activation='relu')(decoded)

    decoded = Dense(units=784,activation='sigmoid')(decoded)





    autoencoder = Model(input_img,decoded)

    encoder = Model(input_img,encoded)

    autoencoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    ## we dont need to compile the encoder model

    ## it is the sub model of the auto encoder



    print(autoencoder.summary())

    print(encoder.summary())

    autoencoder.fit(x_train_re,x_train_re,epochs=epochs,batch_size=250,validation_data=(x_test_re,x_test_re))

    

    encoded_imgs=encoder.predict(x_test_re)

    predicted = autoencoder.predict(x_test_re)

    return encoded_imgs,predicted
def plotting(x_test,encoded_imgs,predicted):

  plt.figure(figsize=(40, 4))

  for i in range(10):

      # display original images

      ax = plt.subplot(3, 20, i + 1)

      plt.imshow(x_test[i].reshape(28, 28))

      plt.gray()

      ax.get_xaxis().set_visible(False)

      ax.get_yaxis().set_visible(False)



      # display encoded images

      ax = plt.subplot(3, 20, i + 1 + 20)

      plt.imshow(encoded_imgs[i].reshape(8,4))

      plt.gray()

      ax.get_xaxis().set_visible(False)

      ax.get_yaxis().set_visible(False)



  # display reconstructed images

      ax = plt.subplot(3, 20, 2*20 +i+ 1)

      plt.imshow(predicted[i].reshape(28, 28))

      plt.gray()

      ax.get_xaxis().set_visible(False)

      ax.get_yaxis().set_visible(False)
def main():

    x_train,x_test = load_images()

    x_train,x_test = normalize(x_train,x_test)

    train_shape,test_shape = get_shape(x_train,x_test)

    print (train_shape)

    print(test_shape)

    x_train_re,x_test_re = reshape_data(x_train,x_test)

    print(x_train_re.shape)

    print(x_test_re.shape)

    encoded_imgs,predicted = encoder(x_train_re,x_test_re)

    plotting(x_test_re,encoded_imgs,predicted)
main()