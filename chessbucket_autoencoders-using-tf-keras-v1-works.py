import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)
# let's prepare our input data. We are using MNIST digits and we are disregrading the labels (since we are only interested in encoding/decoding the input images)

# load the data

(x_train, _), (x_test, _) = load_data('../input/mnist.npz')
x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)

print(x_test.shape)
# we will start simple with a single fully-connected neural layer as encoder and decoder

# this is the siez of our encoded representations

ENCODING_DIM = 32



# input placeholder

input_img = tf.keras.layers.Input(shape=(784,))



# this is the encoded representation of the input

encoded = tf.keras.layers.Dense(ENCODING_DIM, activation='relu')(input_img)



# this is the loss reconstruction of the input

decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)



# this model maps an input to its recommendation

autoencoder = tf.keras.models.Model(input_img, decoded)
# let's also create a seprate encoder model

# this mode maps an input to its encoded representation

encoder = tf.keras.models.Model(input_img, encoded)
# as well as decoder model

# create a placeholder for an encoded (32-dimensional) input

encoded_input = tf.keras.layers.Input(shape=(ENCODING_DIM,))



# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-1]



# create the decoder model

decoder = tf.keras.models.Model(encoded_input, decoder_layer(encoded_input))
# Now let's train our autoencoder to reconstruct MNIST digits

# first we will configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# now let's train our autoencoder for 50 epochs

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
# after 50 epochs the autoencoder seems to reach a stable train/test loss value of about 0.11. We can try to visualize the reconstructed inputs and the encoded representations. We will be using Matplotlib

# encode and decode some digits

# note that we take them from the "test" set

encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)
# now using Matplotlib to plot the images

n = 10 # how many images we will display

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