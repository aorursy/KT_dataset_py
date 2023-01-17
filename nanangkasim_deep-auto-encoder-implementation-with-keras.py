#import library

from keras.datasets import fashion_mnist

from keras.layers import Input, Dense

from keras.models import Model
#Load Dataset

(x_train, _), (x_test, _) = fashion_mnist.load_data()
#Rescale  dataset

import numpy as np



x_train = x_train.astype('float32') / 255.

x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#Build Autoencoder Model



target_dimension = 16



#Encoder

input_img = Input(shape=(784,))

encoder = Dense(128, activation='relu')(input_img)

encoder = Dense(64, activation='relu')(encoder)

encoder = Dense(32, activation='relu')(encoder)



#code

coded = Dense(target_dimension, activation='relu')(encoder)



#Decoder

decoder = Dense(32, activation='relu')(coded)

decoder = Dense(64, activation='relu')(decoder)

decoder = Dense(128, activation='relu')(decoder)

decoder = Dense(784, activation='sigmoid')(decoder)



autoencoder = Model(input_img, decoder)
#compile model

autoencoder.compile(loss = 'binary_crossentropy',

                    optimizer = 'adam')
autoencoder.summary()
#Training model

autoencoder.fit(x_train, x_train,

                epochs=20,

                batch_size=100,

                shuffle=True,

                validation_data=(x_test, x_test))
#Display original data and reconstruction data

import matplotlib.pyplot as plt

decoded_imgs = autoencoder.predict(x_test)



n = 10

plt.figure(figsize=(25, 5))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i+1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i+1 + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

plt.show()