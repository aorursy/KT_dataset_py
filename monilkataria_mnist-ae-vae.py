import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



import keras

from keras.models import Model

from keras.layers import *

from keras import optimizers

from keras import backend as K
# mnist data

df_train = pd.read_csv('../input/mnist_train.csv')

df_test = pd.read_csv('../input/mnist_test.csv')



#Normalization

X_train = df_train.iloc[:, 1:785]

X_train = X_train.values.astype('float32')/255.

output_X_train = X_train.reshape(-1,28,28,1)



X_test = df_test.iloc[:, 1:785]

X_test = X_test.values.astype('float32')/255.

output_X_test = X_test.reshape(-1,28,28,1)



print(X_train.shape, X_test.shape)
#encoder

encoder_inputs = Input(shape = (28,28,1))

 

conv1 = Conv2D(16, (3,3), activation = 'relu', padding = "SAME")(encoder_inputs)

pool1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv1)

conv2 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(pool1)

pool2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv2)

flat = Flatten()(pool2)

 

encoder_outputs = Dense(32, activation = 'relu')(flat)



#sparsity constraint

#enocder_outputs = Dense(32, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(flat)
#AE decoder

dense_layer_d = Dense(7*7*32, activation = 'relu')(encoder_outputs)

output_from_d = Reshape((7,7,32))(dense_layer_d)

conv1_1 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(output_from_d)

upsampling_1 = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(conv1_1)

upsampling_2 = Conv2DTranspose(16, 3, padding='same', activation='relu', strides=(2, 2))(upsampling_1)

decoded_outputs = Conv2DTranspose(1, 3, padding='same', activation='relu')(upsampling_2)



#AE

autoencoder = Model(encoder_inputs, decoded_outputs)



m = 256

n_epoch = 10

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(output_X_train,output_X_train, epochs=n_epoch, batch_size=m, shuffle=True)
test_imgs = autoencoder.predict(output_X_test)
print(test_imgs.shape)

import matplotlib.pyplot as plt



n = 10  # how many digits we will display

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(output_X_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(test_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
#VAE

input_to_z = Dense(32, activation = 'relu')(flat)



latent_dim = 2 # dimension of latent variable

mu = Dense(latent_dim, name='mu')(input_to_z)

sigma = Dense(latent_dim, name='log_var')(input_to_z)

 

encoder_vae = Model(encoder_inputs, mu)



# create latent distribution function and generate vectors

def sampling(args):

    mu, sigma = args

    epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dim),

                              mean=0., stddev=1.)

    return mu + K.exp(sigma) * epsilon

 

z = Lambda(sampling)([mu, sigma])



#create decoder network which is reverse of encoder

decoder_inputs = Input(K.int_shape(z)[1:])

dense_layer_d = Dense(7*7*32, activation = 'relu')(decoder_inputs)

output_from_z_d = Reshape((7,7,32))(dense_layer_d)

trans1_d = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(output_from_z_d)

trans1_1_d = Conv2DTranspose(16, 3, padding='same', activation='relu', strides=(2, 2))(trans1_d)

trans2_d = Conv2DTranspose(1, 3, padding='same', activation='relu')(trans1_1_d)

 

decoder_vae = Model(decoder_inputs, trans2_d)

z_decoded = decoder_vae(z)
#calculate reconstruction loss and KL divergence



class calc_output_with_loss(keras.layers.Layer):



    def vae_loss(self, x, z_decoded):

        x = K.flatten(x)

        z_decoded = K.flatten(z_decoded)



        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)



        kl_loss = -5e-4 * K.mean(1 + sigma - K.square(mu) - K.exp(sigma), axis=-1)

        return K.mean(xent_loss + kl_loss)



    def call(self, inputs):

        x = inputs[0]

        z_decoded = inputs[1]

        loss = self.vae_loss(x, z_decoded)

        self.add_loss(loss, inputs=inputs)

        return x



decoded_outputs_vae = calc_output_with_loss()([encoder_inputs, z_decoded])



# define variational autoencoder model and train it



vae = Model(encoder_inputs, decoded_outputs_vae)

m = 256

n_epoch = 10

vae.compile(optimizer='adam', loss=None)

vae.fit(output_X_train, epochs=n_epoch, batch_size=m, shuffle=True, validation_data=(output_X_test, None))
n = 15  # figure with 15x15 digits

 

digit_size = 28

figure = np.zeros((digit_size * n, digit_size * n))

 

grid_x = np.linspace(-1, 1, n)

grid_y = np.linspace(-1, 1, n)

 

for i, yi in enumerate(grid_x):

    for j, xi in enumerate(grid_y):

        z_sample = np.array([[xi, yi]]) * 1.

        x_decoded = decoder_vae.predict(z_sample)

 

        digit = x_decoded[0].reshape(digit_size, digit_size)

        figure[i * digit_size: (i + 1) * digit_size,

               j * digit_size: (j + 1) * digit_size] = digit

 

plt.figure(figsize=(10, 10))

plt.imshow(figure)

plt.show()