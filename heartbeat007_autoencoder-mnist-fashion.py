! pip install tensorflow-gpu
import numpy as np

import pandas as pd

import tensorflow as tf

import keras

from keras.datasets import fashion_mnist

from keras.models import Model

from keras.layers import Dense,Input
def load_image():

    (x_train,_),(x_test,_) = fashion_mnist.load_data()

    return x_train,x_test
x_train,x_test = load_image()
x_train = tf.keras.utils.normalize(x_train)

x_test = tf.keras.utils.normalize(x_test)
import matplotlib.pyplot as plt

plt.imshow(x_train[0])
print(x_train.shape)

print(x_test.shape)
## reshape data

x_train = x_train.reshape(len(x_train),np.prod(x_train.shape[1:]))

x_test = x_test.reshape(len(x_test),np.prod(x_test.shape[1:]))
print(x_train.shape)

print(x_test.shape)
def encoder_model(x_train,x_test,epochs=1):

    input_img =Input(shape=(784,))

    encoded = Dense(units=128,activation='relu')(input_img)

    encoded = Dense(units=64,activation='relu')(encoded)

    encoded = Dense(units=32,activation='relu')(encoded) ## middle

    decoded = Dense(units=64,activation='relu')(encoded)

    decoded = Dense(units=128,activation='relu')(decoded)

    decoded = Dense(units=784,activation='sigmoid')(decoded)

    

    autoencoder = Model(input_img,decoded)

    encoder = Model(input_img,encoded)

    autoencoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    print(autoencoder.summary())

    print(encoder.summary())

    autoencoder.fit(x_train,x_train,epochs=epochs,batch_size=250,validation_data=(x_test,x_test))

    encoded_imgs = encoder.predict(x_test)

    predicted = autoencoder.predict(x_test)

    return encoded_imgs,predicted
encoded_imgs,predicted = encoder_model(x_train,x_test,100)
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
plotting(x_test,encoded_imgs,predicted)