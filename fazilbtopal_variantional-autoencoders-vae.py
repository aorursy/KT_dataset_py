import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras as kr

%matplotlib inline
# MNIST Dataset parameters.

num_features = 784 # data features (img shape: 28*28).



# Training parameters.

batch_size = 128

epochs = 50



# Network Parameters

hidden_1 = 128 # 1st layer num features.

hidden_2 = 64 # 2nd layer num features (the latent dim).
from tensorflow.keras.datasets import mnist, fashion_mnist



def load_data(choice='mnist', labels=False):

    if choice not in ['mnist', 'fashion_mnist']:

        raise ('Choices are mnist and fashion_mnist')

    

    if choice is 'mnist':

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    else:

        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    

    X_train, X_test = X_train / 255., X_test / 255.

    X_train, X_test = X_train.reshape([-1, 784]), X_test.reshape([-1, 784])

    X_train = X_train.astype(np.float32, copy=False)

    X_test = X_test.astype(np.float32, copy=False)

    

    if labels:

        return (X_train, y_train), (X_test, y_test)

    

    return X_train, X_test





def plot_predictions(y_true, y_pred):    

    f, ax = plt.subplots(2, 10, figsize=(15, 4))

    for i in range(10):

        ax[0][i].imshow(np.reshape(y_true[i], (28, 28)), aspect='auto')

        ax[1][i].imshow(np.reshape(y_pred[i], (28, 28)), aspect='auto')

    plt.tight_layout()
def plot_digits(X, y, encoder, batch_size=128):

    """Plots labels and MNIST digits as function of 2D latent vector



    Parameters:

    ----------

    encoder: Model

        A Keras Model instance

    X: np.ndarray

        Test data

    y: np.ndarray

        Test data labels

    batch_size: int

        Prediction batch size

    """

    # display a 2D plot of the digit classes in the latent space

    z_mean, _, _ = encoder.predict(X, batch_size=batch_size)

    plt.figure(figsize=(12, 10))

    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y)

    plt.colorbar()

    plt.xlabel("z[0] Latent Dimension")

    plt.ylabel("z[1] Latent Dimension")

    plt.show()

    

    

def generate_manifold(decoder):

    """Generates a manifold of MNIST digits from a random noisy data.



    Parameters:

    ----------

    decoder: Model

        A Keras Model instance

    """

    

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

    plt.xlabel("z[0] Latent Dimension")

    plt.ylabel("z[1] Latent Dimension")

    plt.imshow(figure, cmap='Greys_r')

    plt.show()
inputs = kr.Input(shape=(num_features, ))

encoder = kr.layers.Dense(hidden_1, activation='sigmoid')(inputs)

encoder = kr.layers.Dense(hidden_2, activation='sigmoid')(encoder)

encoder_model = kr.Model(inputs, encoder, name='encoder')

encoder_model.summary()
latent_dim = kr.Input(shape=(hidden_2, ))

decoder = kr.layers.Dense(hidden_1, activation='sigmoid')(latent_dim)

decoder = kr.layers.Dense(num_features, activation='sigmoid')(decoder)

decoder_model = kr.Model(latent_dim, decoder, name='decoder')

decoder_model.summary()
outputs = decoder_model(encoder_model(inputs))

mnist_model = kr.Model(inputs, outputs )

mnist_model.compile(optimizer='adam', loss='mse')

mnist_model.summary()
X_train, X_test = load_data('mnist')

mnist_model.fit(x=X_train, y=X_train, batch_size=batch_size, shuffle=False, epochs=epochs)
y_true = X_test[:10]

y_pred = mnist_model.predict(y_true)

plot_predictions(y_true, y_pred)
# Encoder

inputs = kr.Input(shape=(num_features, ))

encoder = kr.layers.Dense(hidden_1, activation='sigmoid')(inputs)

encoder = kr.layers.Dense(hidden_2, activation='sigmoid')(encoder)

encoder_model = kr.Model(inputs, encoder, name='encoder')

encoder_model.summary()



# Decoder

latent_dim = kr.Input(shape=(hidden_2, ))

decoder = kr.layers.Dense(hidden_1, activation='sigmoid')(latent_dim)

decoder = kr.layers.Dense(num_features, activation='sigmoid')(decoder)

decoder_model = kr.Model(latent_dim, decoder, name='decoder')

decoder_model.summary()



# AE

outputs = decoder_model(encoder_model(inputs))

fmnist_model = kr.Model(inputs, outputs )

fmnist_model.compile(optimizer='adam', loss='mse')

fmnist_model.summary()
X_train, X_test = load_data('fashion_mnist')

fmnist_model.fit(x=X_train, y=X_train, batch_size=batch_size, shuffle=False, epochs=epochs)
y_true = X_test[:10]

y_pred = fmnist_model.predict(y_true)

plot_predictions(y_true, y_pred)
def sampling(args):

    """Reparameterization trick. Instead of sampling from Q(z|X), 

    sample eps = N(0,I) z = z_mean + sqrt(var)*eps.



    Parameters:

    -----------

    args: list of Tensors

        Mean and log of variance of Q(z|X)



    Returns

    -------

    z: Tensor

        Sampled latent vector

    """



    z_mean, z_log_var = args

    eps = tf.random_normal(tf.shape(z_log_var), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')

    z = z_mean + tf.exp(z_log_var / 2) * eps

    return z
hidden_dim = 512

latent_dim = 2  # The bigger this is, more accurate the network is but 2 is for illustration purposes.
inputs = kr.layers.Input(shape=(num_features, ), name='input')

x = kr.layers.Dense(hidden_dim, activation='relu')(inputs)

z_mean = kr.layers.Dense(latent_dim, name='z_mean')(x)

z_log_var = kr.layers.Dense(latent_dim, name='z_log_var')(x)
z = kr.layers.Lambda(sampling, name='z')([z_mean, z_log_var])



# instantiate encoder model

encoder = kr.Model(inputs, [z_mean, z_log_var, z], name='encoder')

encoder.summary()
latent_inputs = kr.layers.Input(shape=(latent_dim,), name='z_sampling')

x = kr.layers.Dense(hidden_dim, activation='relu')(latent_inputs)

outputs = kr.layers.Dense(num_features, activation='sigmoid')(x)



# instantiate decoder model

decoder = kr.Model(latent_inputs, outputs, name='decoder')

decoder.summary()
# # VAE model = encoder + decoder

outputs = decoder(encoder(inputs)[2])  # Select the Z value from outputs of the encoder

vae = kr.Model(inputs, outputs, name='vae')
# Reconstruction loss

reconstruction_loss = tf.losses.mean_squared_error(inputs, outputs)

reconstruction_loss = reconstruction_loss * num_features



# KL Divergence loss

kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)

kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)

vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)



vae.add_loss(vae_loss)

vae.compile(optimizer='adam')

vae.summary()
(X_train, _),  (X_test, y) = load_data('mnist', labels=True)

vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))
generate_manifold(decoder)
plot_digits(X_test, y, encoder)  # y for label coloring
# new network parameters

input_shape = (28, 28, 1)

batch_size = 128

filters = 32

latent_dim = 2

epochs = 30
inputs = kr.Input(shape=input_shape, name='input')



x = kr.layers.Conv2D(filters, (3, 3), activation='relu', strides=2, padding='same')(inputs)

x = kr.layers.Conv2D(filters*2, (3, 3), activation='relu', strides=2, padding='same')(x)



# shape info needed to build decoder model

shape = x.get_shape().as_list()



# generate latent vector Q(z|X)

x = kr.layers.Flatten()(x)

x = kr.layers.Dense(16, activation='relu')(x)

z_mean = kr.layers.Dense(latent_dim, name='z_mean')(x)

z_log_var = kr.layers.Dense(latent_dim, name='z_log_var')(x)



z = kr.layers.Lambda(sampling, name='z')([z_mean, z_log_var])



# instantiate encoder model

encoder = kr.Model(inputs, [z_mean, z_log_var, z], name='encoder')

encoder.summary()
# build decoder model

latent_inputs = kr.Input(shape=(latent_dim,), name='z_sampling')

x = kr.layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)

x = kr.layers.Reshape((shape[1], shape[2], shape[3]))(x)

x = kr.layers.Conv2DTranspose(filters*2, (3, 3), activation='relu', strides=2, padding='same')(x)

x = kr.layers.Conv2DTranspose(filters, (3, 3), activation='relu', strides=2, padding='same')(x)

outputs = kr.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)



# instantiate decoder model

decoder = kr.Model(latent_inputs, outputs, name='decoder')

decoder.summary()
# instantiate VAE model

outputs = decoder(encoder(inputs)[2])

vae = kr.Model(inputs, outputs, name='vae')
# Reconstruction loss

reconstruction_loss = tf.losses.mean_squared_error(inputs, outputs)

reconstruction_loss = reconstruction_loss * num_features



# KL Divergence loss

kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)

kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)

vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)



vae.add_loss(vae_loss)

vae.compile(optimizer='adam')

vae.summary()
X_train, X_test = X_train.reshape([-1, 28, 28, 1]), X_test.reshape([-1, 28, 28, 1])

vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))
generate_manifold(decoder)
plot_digits(X_test, y, encoder)