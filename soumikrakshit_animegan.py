import os, warnings

warnings.filterwarnings('ignore')
import numpy as np

import cv2, glob

from random import shuffle, randint

from tqdm import tqdm

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Reshape, BatchNormalization

from tensorflow.keras.layers import Activation, Conv2DTranspose, Conv2D, LeakyReLU

from tensorflow.keras.optimizers import Adam

from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG
image_files = glob.glob('../input/data/data/*.png')

shuffle(image_files)
x = []

for file in tqdm(image_files):

    image = cv2.imread(file)

    image = image / 127.5

    image = image - 1

    x.append(image)

x = np.array(x)

x.shape
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (16, 16))

plt.setp(axes.flat, xticks = [], yticks = [])

for i, ax in enumerate(axes.flat):

    index = randint(0, 10000)

    ax.imshow(x[index], cmap = 'gray')

plt.show()
def build_discriminator(image_shape, learning_rate, beta_1):

    discriminator = Sequential([

        Conv2D(

            filters = 64,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform',

            input_shape = (image_shape)

        ),

        LeakyReLU(0.2),

        

        Conv2D(

            filters = 128,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform',

        ),

        BatchNormalization(momentum = 0.5),

        LeakyReLU(0.2),

        

        Conv2D(

            filters = 256,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform',

        ),

        BatchNormalization(momentum = 0.5),

        LeakyReLU(0.2),

        

        Conv2D(

            filters = 512,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform',

        ),

        BatchNormalization(momentum = 0.5),

        LeakyReLU(0.2),

        

        Flatten(),

        Dense(1),

        Activation('sigmoid')

    ], name = 'Discriminator')

    

    discriminator.compile(

        loss = 'binary_crossentropy',

        optimizer = Adam(

            lr = learning_rate,

            beta_1 = beta_1

        ),

        metrics = None

    )

    

    return discriminator
def build_generator(input_shape, learning_rate, beta_1):

    generator = Sequential([

        Dense(

            input_shape,

            kernel_initializer = 'glorot_uniform',

            input_shape = (1, 1, 100)

        ),

        Reshape(target_shape = (4, 4, 512)),

        BatchNormalization(momentum = 0.5),

        Activation('relu'),

        

        Conv2DTranspose(

            filters = 256,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform'

        ),

        BatchNormalization(momentum = 0.5),

        Activation('relu'),

        

        Conv2DTranspose(

            filters = 128,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform'

        ),

        BatchNormalization(momentum = 0.5),

        Activation('relu'),

        

        Conv2DTranspose(

            filters = 64,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform'

        ),

        BatchNormalization(momentum = 0.5),

        Activation('relu'),

        

        Conv2DTranspose(

            filters = 3,

            kernel_size = (5, 5),

            strides = (2, 2),

            padding = 'same',

            data_format = 'channels_last',

            kernel_initializer = 'glorot_uniform'

        ),

        Activation('tanh'),

    ], name = 'Generator')

    

    generator.compile(

        loss = 'binary_crossentropy',

        optimizer = Adam(

            lr = learning_rate,

            beta_1 = beta_1

        ),

        metrics = None

    )

    

    return generator
def build_gan(generator, discriminator, learning_rate, beta_1):

    gan = Sequential([

        generator,

        discriminator

    ], name = 'GAN')

    gan.compile(

        loss = 'binary_crossentropy',

        optimizer = Adam(

            lr = learning_rate,

            beta_1 = beta_1

        ),

        metrics = None

    )

    return gan
discriminator = build_discriminator((64, 64, 3), 0.0002, 0.5)

discriminator.summary()
SVG(model_to_dot(discriminator, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
generator = build_generator(np.prod(discriminator.layers[-4].output_shape[1:]), 0.00015, 0.5)

generator.summary()
SVG(model_to_dot(generator, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
discriminator.trainable = False

gan = build_gan(generator, discriminator, 0.00015, 0.5)

gan.summary()
SVG(model_to_dot(gan, show_shapes = True, show_layer_names = True).create(prog = 'dot', format = 'svg'))
EPOCHS = 15000

BATCH_SIZE = 32
def plot_images(nrows, ncols, figsize, generator):

    noise = np.random.normal(0, 1, size = (BATCH_SIZE * 2, ) + (1, 1, 100))

    prediction = generator.predict(noise)

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    plt.setp(axes.flat, xticks = [], yticks = [])

    for i, ax in enumerate(axes.flat):

        index = randint(0, 63)

        ax.imshow(cv2.cvtColor(prediction[index], cv2.COLOR_BGR2RGB), cmap = 'gray')

    plt.show()
discriminator_loss_history, generator_loss_history = [], []



for epoch in tqdm(range(1, EPOCHS + 1)):

    

    # Select a random batch of images from training data

    index = np.random.randint(0, x.shape[0], BATCH_SIZE)

    batch_images = x[index]

    

    # Adversarial Noise

    noise = np.random.normal(0, 1, size = (BATCH_SIZE, ) + (1, 1, 100))

    

    # Fenerate Fake Images

    generated_images = generator.predict(noise)

    

    # Adding noise to the labels before passing to the discriminator

    real_y = (np.ones(BATCH_SIZE) -  np.random.random_sample(BATCH_SIZE) * 0.2)

    fake_y = np.random.random_sample(BATCH_SIZE) * 0.2

    

    # Training the discriminator

    discriminator.trainable = True

    discriminator_loss = discriminator.train_on_batch(batch_images, real_y)

    discriminator_loss += discriminator.train_on_batch(generated_images, fake_y)

    discriminator.trainable = False

    

    # Adversarial Noise

    noise = np.random.normal(0, 1, size = (BATCH_SIZE * 2,) + (1, 1, 100))

    

    # We try to mislead the discriminator by giving the opposite labels

    fake_y = (np.ones(BATCH_SIZE * 2) - np.random.random_sample(BATCH_SIZE * 2) * 0.2)

    

    # Training the Generator

    generator_loss = gan.train_on_batch(noise, fake_y)

    

    if epoch % 100 == 0:

        discriminator_loss_history.append(discriminator_loss)

        generator_loss_history.append(generator_loss)

        if epoch % 1000 == 0:

            plot_images(2, 8, (16, 4), generator)
plt.figure(figsize = (20, 8))

plt.plot(generator_loss_history)

plt.title('Generator Loss History')

plt.show()
plt.figure(figsize = (20, 8))

plt.plot(discriminator_loss_history)

plt.title('Discriminator Loss History')

plt.show()
plot_images(4, 4, (16, 16), generator)
generator.save('./generator.h5')