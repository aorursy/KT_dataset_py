# -*- coding: utf-8 -*-

"""

Created on Wed Nov  9 10:46:55 2016



@author: Murasaki

"""



from keras.layers import Input, Dense

from keras.models import Model



# this is the size of our encoded representations

encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats



# this is our input placeholder

input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input

decoded = Dense(784, activation='sigmoid')(encoded)



# this model maps an input to its reconstruction

autoencoder = Model(input=input_img, output=decoded)



# this model maps an input to its encoded representation

encoder = Model(input=input_img, output=encoded)



# create a placeholder for an encoded (32-dimensional) input

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-1]

# create the decoder model

decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))



autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')





import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split







x = pd.read_csv('../input/train.csv')



x_train, x_test = train_test_split(x, test_size=0.2)



x_train = x_train.as_matrix()

x_test = x_test.as_matrix()



x_train = x_train[0:784]

x_test = x_test[0:784]





autoencoder.fit(x_train, x_train,

                nb_epoch=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test, x_test))



# encode and decode some digits

# note that we take them from the *test* set

encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)



# use Matplotlib (don't ask)

import matplotlib.pyplot as plt



n = 10  # how many digits we will display

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()