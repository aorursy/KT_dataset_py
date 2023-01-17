!pip install tensorflow-gpu

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import keras 

from keras.models import Model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation

from keras.datasets import cifar10
def load_images():

    (x_train,_),(x_test,_)  = cifar10.load_data()

    return x_train,x_test
x_train,x_test = load_images()
def test_plot(x):

    plt.imshow(x[0])
test_plot(x_train)
test_plot(x_test)
def normalize(x_train,x_test):

    x_train = keras.utils.normalize(x_train)

    x_test = keras.utils.normalize(x_test)

    return x_train,x_test
#x_train_1,x_test_1 = normalize(x_train,x_test)

#x_train_re,x_test_re = normalize(x_train,x_test)

x_train_re,x_test_re = x_train,x_test
test_plot(x_train_re)
print(x_train_re.shape[1:])

print(x_test_re.shape)

input_shape = x_train.shape[1:]

receptive_field=(3,3)

pooling_field = (2,2)
def CONVautoencoder(x_train_re,x_test_re,epochs=200):

    ## functional approach

    ## input dimension is 3 so the output dimension will be 3

    



    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(64, (3, 3), padding='same')(input_img)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)



    x = Conv2D(16, (3, 3), padding='same')(encoded)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(3, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)

    decoded = Activation('sigmoid')(x)

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







    return encoded_imgs,predicted

    
def plotting(x_test,encoded_imgs,predicted):

  plt.figure(figsize=(40, 4))

  for i in range(10):

      # display original images

      ax = plt.subplot(3, 20, i + 1)

      plt.imshow(x_test[i].reshape(32, 32,3))

      #plt.gray()

      ax.get_xaxis().set_visible(False)

      ax.get_yaxis().set_visible(False)







  # display reconstructed images

      ax = plt.subplot(3, 20, 2*20 +i+ 1)

      plt.imshow(predicted[i].reshape(32, 32,3))

      #plt.gray()

      ax.get_xaxis().set_visible(False)

      ax.get_yaxis().set_visible(False)
def main():

    x_train,x_test = load_images()

    x_train,x_test = normalize(x_train,x_test)

   

    #x_train_re,x_test_re = reshape_data(x_train,x_test)

    #print(x_train_re.shape)

    #print(x_test_re.shape)

    encoded_imgs,predicted = CONVautoencoder(x_train_re,x_test_re)

    plotting(x_test_re,encoded_imgs,predicted)
main()