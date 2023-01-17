import warnings

warnings.filterwarnings('ignore')
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam

from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

from tqdm import tqdm
IMAGE_WIDTH = 28

IMAGE_HEIGHT = 28

IMAGE_CHANNELS = 1

BATCH_SIZE = 128

LATENT_DIMENSION = 100

IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

EPOCHS = 8000
def load_data():

    (x_train, _), (_, _) = mnist.load_data()

    x_train = x_train / 127.5 - 1.

    x_train = np.expand_dims(x_train, axis = 3)

    return x_train
x_train = load_data()

x_train.shape
def build_generator(latent_dimension, optimizer):

    generator = Sequential([

        Dense(256, input_dim = latent_dimension, activation = 'tanh'),

        Dense(128 * 7 * 7),

        BatchNormalization(),

        Activation('tanh'),

        Reshape((7, 7, 128)),

        UpSampling2D(size = (2, 2)),

        Conv2D(64, (5, 5), padding = 'same', activation = 'tanh'),

        UpSampling2D(size = (2, 2)),

        Conv2D(1, (5, 5), padding = 'same', activation = 'tanh')

    ])

    generator.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    return generator
def build_discriminator(image_shape, optimizer):

    discriminator = Sequential([

        Conv2D(64, (5, 5), padding = 'same', input_shape = image_shape, activation = 'tanh'),

        MaxPooling2D(pool_size = (2, 2)),

        Conv2D(128, (5, 5), activation = 'tanh'),

        MaxPooling2D(pool_size = (2, 2)),

        Flatten(),

        Dense(1024, activation = 'tanh'),

        Dense(1, activation = 'sigmoid')

    ])

    discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    return discriminator
def build_gan(generator, discriminator, latent_dimension, optimizer):

    discriminator.trainable = False

    gan_input = Input(shape = (latent_dimension, ))

    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(gan_input, gan_output, name = 'GAN')

    gan.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return gan
optimizer = Adam(0.0002, 0.5)
generator = build_generator(LATENT_DIMENSION, optimizer)

generator.summary()
SVG(model_to_dot(generator, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
discriminator = build_discriminator(IMAGE_SHAPE, optimizer)

discriminator.summary()
SVG(model_to_dot(discriminator, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
gan = build_gan(generator, discriminator, LATENT_DIMENSION, optimizer)

gan.summary()
SVG(model_to_dot(gan, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
def plot_images(nrows, ncols, figsize, generator):

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    plt.setp(axes.flat, xticks = [], yticks = [])

    noise = np.random.normal(0, 1, (nrows * ncols, LATENT_DIMENSION))

    generated_images = generator.predict(noise).reshape(nrows * ncols, IMAGE_WIDTH, IMAGE_HEIGHT)

    for i, ax in enumerate(axes.flat):

        ax.imshow(generated_images[i], cmap = 'gray')

    plt.show()
generator_loss_history, discriminator_loss_history = [], []



for epoch in tqdm(range(1, EPOCHS + 1)):

    

    # Select a random batch of images from training data

    index = np.random.randint(0, x_train.shape[0], BATCH_SIZE)

    batch_images = x_train[index]

    

    # Adversarial Noise

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIMENSION))

    

    # Generate fake images

    generated_images = generator.predict(noise)

    

    # Construct batches of real and fake data

    x = np.concatenate([batch_images, generated_images])

    

    # Labels for training the discriminator

    y_discriminator = np.zeros(2 * BATCH_SIZE)

    y_discriminator[: BATCH_SIZE] = 0.9

    

    # train the discrimator to distinguish between fake data and real data

    discriminator.trainable = True

    discriminator_loss = discriminator.train_on_batch(x, y_discriminator)

    discriminator_loss_history.append(discriminator_loss)

    discriminator.trainable = False

    

    # Training the GAN

    generator_loss = gan.train_on_batch(noise, np.ones(BATCH_SIZE))

    generator_loss_history.append(generator_loss)

    

    if epoch % 1000 == 0:

        plot_images(1, 8, (16, 4), generator)
plot_images(2, 8, (16, 6), generator)
generator.save('./generator.h5')