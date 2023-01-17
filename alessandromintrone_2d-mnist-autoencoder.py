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
input_img = tf.keras.layers.Input(shape=(28, 28, 1)) # adapt this if using `channels_first` image data format



x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)



# at this point the representation is (4, 4, 8) i.e. 128-dimensional



x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)

x = tf.keras.layers.UpSampling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = tf.keras.layers.UpSampling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)

x = tf.keras.layers.UpSampling2D((2, 2))(x)

decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)



autoencoder = tf.keras.models.Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# to train this model we will with original MNIST digits with shape (samples, 3, 28, 28) and we will just normalize pixel values between 0 and 1

(x_train, _), (x_test, _) = load_data('../input/mnist.npz')



print(x_train.shape)
x_train = x_train.astype('float32') / 255.

x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#%load_ext tensorboard.note

#%tensorboard --logdir ./tmp/autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test), callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./tmp/autoencoder')])
decoded_imgs = autoencoder.predict(x_test)



n = 10

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

 # we can also look at the 128-dimensional encoded representations. These representations are 8x4x4, so we reshape them to 4x32 in order to be able to display them as grayscale images

 

n = 10

encoder = tf.keras.models.Model(input_img, encoded)

encoded_imgs = encoder.predict(x_test)

plt.figure(figsize=(20, 8))

for i in range(n):

     ax = plt.subplot(1, n, i + 1)

     plt.imshow(encoded_imgs[i + 1].reshape(4, 4 * 8).T)

     plt.gray()

     ax.get_xaxis().set_visible(False)

     ax.get_yaxis().set_visible(False)

plt.show()
# helper functions



# reparameterization trick

# instead of sampling from Q(z|X), sample eps = N(0,I)

# z = z_mean + sqrt(var)*eps

def sampling(args):

    z_mean, z_log_var = args

    batch = tf.keras.backend.shape(z_mean)[0]

    dim = tf.keras.backend.int_shape(z_mean)[1]

    # by default, random_normal has mean=0 and std=1.0

    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

    

    

def plot_results(models, data, batch_size=128, model_name="vae_mnist"):

    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:

        models (tuple): encoder and decoder models

        data (tuple): test data and label

        batch_size (int): prediction batch size

        model_name (string): which model is using this function

    """



    encoder, decoder = models

    x_test, y_test = data

    os.makedirs(model_name, exist_ok=True)



    filename = os.path.join(model_name, "vae_mean.png")

    # display a 2D plot of the digit classes in the latent space

    z_mean, _, _ = encoder.predict(x_test,

                                   batch_size=batch_size)

    plt.figure(figsize=(12, 10))

    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)

    plt.colorbar()

    plt.xlabel("z[0]")

    plt.ylabel("z[1]")

    plt.savefig(filename)

    plt.show()



    filename = os.path.join(model_name, "digits_over_latent.png")

    # display a 30x30 2D manifold of digits

    n = 30

    digit_size = 28

    figure = np.zeros((digit_size * n, digit_size * n))

    # linearly spaced coordinates corresponding to the 2D plot

    # of digit classes in the latent space

    grid_x = np.linspace(-4, 4, n)

    grid_y = np.linspace(-4, 4, n)[::-1]



    for i, yi in enumerate(grid_y):

        for j, xi in enumerate(grid_x):

            z_sample = np.array([[xi, yi]])

            x_decoded = decoder.predict(z_sample)

            digit = x_decoded[0].reshape(digit_size, digit_size)

            figure[i * digit_size: (i + 1) * digit_size,

                   j * digit_size: (j + 1) * digit_size] = digit



    plt.figure(figsize=(10, 10))

    start_range = digit_size // 2

    end_range = n * digit_size + start_range + 1

    pixel_range = np.arange(start_range, end_range, digit_size)

    sample_range_x = np.round(grid_x, 1)

    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)

    plt.yticks(pixel_range, sample_range_y)

    plt.xlabel("z[0]")

    plt.ylabel("z[1]")

    plt.imshow(figure, cmap='Greys_r')

    plt.savefig(filename)

    plt.show()
#Import and preprocess DENSE

(x_train, y_train), (x_test, y_test) = load_data('../input/mnist.npz')



image_size = x_train.shape[1]

original_dim = image_size * image_size

x_train = np.reshape(x_train, [-1, original_dim])

x_test = np.reshape(x_test, [-1, original_dim])

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255
#Network parameters DENSE

input_shape = (original_dim, )

intermediate_dim = 512

batch_size = 128

latent_dim = 2

epochs = 50
# build encoder model DENSE

inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')

x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)

z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)

z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)



# use reparameterization trick to push the sampling out as input

z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])



# instantiate encoder model

encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

encoder.summary()

# tf.keras.utils.plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)



# build decoder model

latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')

x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)

outputs = tf.keras.layers.Dense(original_dim, activation='sigmoid')(x)



# instantiate decoder model

decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

decoder.summary()

# tf.keras.utils.plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)



# instantiate VAE model

outputs = decoder(encoder(inputs)[2])

vae = tf.keras.models.Model(inputs, outputs, name='vae_mlp')



models = (encoder, decoder)

data = (x_test, y_test)

# reconstruction_loss = tf.keras.losses.mse(inputs, outputs)

reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)



reconstruction_loss *= original_dim



kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)

kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)

kl_loss *= -0.5

vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

vae.compile(optimizer='adam')

vae.summary()

# tf.keras.utils.plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

vae.save_weights('vae_mlp.mnist.h5')
# Because the VAE is a generative model, we can also use it to generate new digits! Here we will scan the latent plane, sampling latent points at regular intervals and generating the corresponding digit for each of these points. This gives us a visualization of the latent manifold that "generates" the MNIST digits.

plot_results(models, data, batch_size=batch_size, model_name='vae_mlp')
# helper functions



# reparameterization trick

# instead of sampling from Q(z|X), sample eps = N(0,I)

# z = z_mean + sqrt(var)*eps

def sampling(args):

    z_mean, z_log_var = args

    batch = tf.keras.backend.shape(z_mean)[0]

    dim = tf.keras.backend.int_shape(z_mean)[1]

    # by default, random_normal has mean=0 and std=1.0

    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

    

    

def plot_results(models, data, batch_size=128, model_name="vae_mnist"):

    

    from IPython.display import FileLink



    encoder, decoder = models

    x_test, y_test = data

    os.makedirs(model_name, exist_ok=True)



    filename = os.path.join(model_name, "vae_mean.png")

    FileLink(filename)

    

    # display a 2D plot of the digit classes in the latent space

    z_mean, _, _ = encoder.predict(x_test,

                                   batch_size=batch_size)

    plt.figure(figsize=(12, 10))

    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)

    plt.colorbar()

    plt.xlabel("z[0]")

    plt.ylabel("z[1]")

    plt.show()

    plt.savefig(filename)

    

    filename = os.path.join(model_name, "digits_over_latent.png")

    FileLink(filename)

    # display a 30x30 2D manifold of digits

    n = 30

    digit_size = 28

    figure = np.zeros((digit_size * n, digit_size * n))

    # linearly spaced coordinates corresponding to the 2D plot

    # of digit classes in the latent space

    grid_x = np.linspace(-2, 2, n)

    grid_y = np.linspace(-2, 2, n)[::-1]



    for i, yi in enumerate(grid_y):

        for j, xi in enumerate(grid_x):

            z_sample = np.array([[xi, yi]])

            x_decoded = decoder.predict(z_sample)

            digit = x_decoded[0].reshape(digit_size, digit_size)

            figure[i * digit_size: (i + 1) * digit_size,

                   j * digit_size: (j + 1) * digit_size] = digit



    plt.figure(figsize=(10, 10))

    start_range = digit_size // 2

    end_range = n * digit_size + start_range + 1

    pixel_range = np.arange(start_range, end_range, digit_size)

    sample_range_x = np.round(grid_x, 1)

    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)

    plt.yticks(pixel_range, sample_range_y)

    plt.xlabel("z[0]")

    plt.ylabel("z[1]")

    plt.imshow(figure, cmap='Greys_r')

    plt.savefig(filename)

    plt.show()
#Import and preprocess CONVOLUTIONAL

(x_train, y_train), (x_test, y_test) = load_data('../input/mnist.npz')





image_size = x_train.shape[1]

#original_dim = image_size * image_size

x_train = np.reshape(x_train, [-1,  image_size, image_size, 1])

x_test = np.reshape(x_test, [-1,  image_size, image_size, 1])

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255



# Binarization

x_train[x_train >= .5] = 1.

x_train[x_train < .5] = 0.

x_test[x_test >= .5] = 1.

x_test[x_test < .5] = 0.



print(x_train.shape)

print(y_train.shape)
#Network parameters CONVOLUTIONAL

input_shape = (image_size, image_size, 1)

batch_size = 256

layers_number = 2

kernel_size = 3

filters = 32

latent_dim = 2

dense_density = 128

kl_weight = -0.75

epochs = 100

opt = tf.keras.optimizers.Adam(lr=0.0015)

load_weights = False
#Encoder CONVOLUTIONAL

inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')

x = inputs

for i in range(layers_number):

    filters *= 2

    x = tf.keras.layers.Conv2D(filters=filters,

               kernel_size=kernel_size,

               activation='relu',

               strides=2,

               padding='same')(x)



# shape info needed to build decoder model

shape = tf.keras.backend.int_shape(x)



# generate latent vector Q(z|X)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(dense_density , activation='relu')(x)

z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)

z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)



# use reparameterization trick to push the sampling out as input

# note that "output_shape" isn't necessary with the TensorFlow backend

z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])



# instantiate encoder model

encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

encoder.summary()
#Decoder CONVOLUTIONAL

latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')

x = tf.keras.layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)

x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)



# use Conv2DTranspose to reverse the conv layers from the encoder

for i in range(layers_number):

    x = tf.keras.layers.Conv2DTranspose(filters=filters,

                                        kernel_size=kernel_size,

                                        activation='relu',

                                        strides=2,

                                        padding='same')(x)

    filters //= 2



outputs = tf.keras.layers.Conv2DTranspose(filters=1,

                                          kernel_size=kernel_size,

                                          activation='sigmoid',

                                          padding='same',

                                          name='decoder_output')(x)



# instantiate decoder model

decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

decoder.summary()
#VAE model CONVOLUTIONAL

outputs = decoder(encoder(inputs)[2])

vae = tf.keras.models.Model(inputs, outputs, name='vae')



#Loss = reconstruction + KL loss

reconstruction_loss = tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(inputs), tf.keras.backend.flatten(outputs))

reconstruction_loss *= image_size * image_size



kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)

kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)

kl_loss *= kl_weight



vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

#vae.compile(optimizer='rmsprop')

#vae.compile(optimizer='adam')

vae.compile(optimizer=opt)

vae.summary()
#Fit model CONVOLUTIONAL

if load_weights:

    vae = vae.load_weights(args.weights)

else:

    # train the autoencoder

    vae.fit(x_train,

            epochs=epochs,

            batch_size=batch_size,

            validation_data=(x_test, None))

    vae.save_weights('vae_cnn_mnist.h5')
#plot

models = (encoder, decoder)

data = (x_test, y_test)

plot_results(models, data, batch_size=batch_size, model_name='vae_cnn')