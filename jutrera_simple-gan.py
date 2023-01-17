import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from pathlib import Path

from PIL import Image

import glob

import numpy as np

import random



path_base = '/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/train'

allfiles = [f for f in glob.glob(path_base + "/**/*.jpg", recursive=True)]

random.shuffle(allfiles)



files = []

for i in range(0, len(allfiles)):

    my_file = Path(allfiles[i])

    if my_file.is_file():

        im = Image.open(my_file)

        image = np.array(im)

        if image.ndim == 3:

            files.append(allfiles[i])

        else:

            print(allfiles[i])



path_base = '/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/test'

allfiles = [f for f in glob.glob(path_base + "/**/*.jpg", recursive=True)]

random.shuffle(allfiles)



for i in range(0, len(allfiles)):

    my_file = Path(allfiles[i])

    if my_file.is_file():

        im = Image.open(my_file)

        image = np.array(im)

        if image.ndim == 3:

            files.append(allfiles[i])

        else:

            print(allfiles[i])



train_df = np.array(files)

train_df.shape
import numpy as np

from scipy import misc

from PIL import Image, ImageOps

import glob

import matplotlib.pyplot as plt

import scipy.misc

from matplotlib.pyplot import imshow

%matplotlib inline

from IPython.display import SVG

import cv2

import seaborn as sn

import pandas as pd

import pickle

from pathlib import Path
from keras import layers

from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout

from keras.layers import Reshape, UpSampling2D, Conv2DTranspose, LeakyReLU

from keras.models import Sequential, Model, load_model

from keras.preprocessing import image

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import decode_predictions

from keras.applications import densenet

from keras.utils import layer_utils, np_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.initializers import glorot_uniform

from keras import losses

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from keras.optimizers import Adam, RMSprop

from keras import regularizers

from keras import backend as K

from keras import datasets

from sklearn.metrics import confusion_matrix, classification_report



import tensorflow as tf
latent_dim = 100

height = 32

width = 32

channels = 3

batch_size = 32
def build_generator():

    generator_input = layers.Input(shape=(latent_dim,))



    # First, transform the input into a 16x16 128-channels feature map

    x = layers.Dense(128 * 16 * 16)(generator_input)

    x = layers.LeakyReLU()(x)

    x = layers.Reshape((16, 16, 128))(x)



    # Then, add a convolution layer

    x = layers.Conv2D(256, 5, padding='same')(x)

    x = layers.LeakyReLU()(x)



    # Upsample to 32x32

    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)

    x = layers.LeakyReLU()(x)



    # Few more conv layers

    x = layers.Conv2D(256, 5, padding='same')(x)

    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 5, padding='same')(x)

    x = layers.LeakyReLU()(x)



    # Produce a 32x32 1-channel feature map

    x = layers.Conv2D(channels, 7, padding='same')(x)

    x = layers.LeakyReLU()(x)

    generator = Model(generator_input, x)

    generator.summary()



    return generator
def build_discriminator():

    discriminator_input = layers.Input(shape=(height, width, channels))

    x = layers.Conv2D(128, 3)(discriminator_input)

    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=2)(x)

    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=2)(x)

    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=2)(x)

    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)



    # One dropout layer - important trick!

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(10)(x)

    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)



    # Classification layer

    x = layers.Dense(1)(x)



    discriminator = Model(discriminator_input, x)

    discriminator.summary()



    return discriminator
def build_gan(discriminator, generator):

    # Set discriminator weights to non-trainable

    # (will only apply to the `gan` model)

    discriminator.trainable = False



    gan_input = layers.Input(shape=(latent_dim,))

    gan_output = discriminator(generator(gan_input))

    gan = Model(gan_input, gan_output)



    return gan
discriminator = build_discriminator()

discriminator_optimizer = RMSprop(lr=0.001, clipvalue=1.0, decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='mse')
generator = build_generator()
gan = build_gan(discriminator, generator)

gan_optimizer = RMSprop(lr=0.0001, clipvalue=1.0, decay=1e-8)

gan.compile(optimizer=gan_optimizer, loss='mse')
from pathlib import Path

from PIL import Image, ImageOps



def generate_real(data, index, size):

    im = Image.open(data[index])

    im = ImageOps.fit(im, size, Image.ANTIALIAS)

    npimage = np.array(im)

    npimage = npimage / 255



    return npimage
def plot_images(save2file=False, fake=True, samples=16, images=None, dpi = 80):

    mul = samples



    plt.figure(figsize=(mul*width/dpi,mul*height/dpi))

    for i in range(samples):

        plt.subplot(4, 4, i+1)

        image = images[i, :, :, :]

        image = np.reshape(image, [width, height, channels])

        plt.imshow((image * 255).astype(np.uint8))

        plt.axis('off')

    plt.tight_layout()

    if save2file:

        plt.savefig(filename)

        plt.close('all')

    else:

        plt.show()
def plot_image(image, save2file=False, dpi=80):

    plt.figure(figsize=(width/dpi,height/dpi))

    image = np.reshape(image, [width, height, channels])

    plt.imshow((image * 255).astype(np.uint8))

    plt.axis('off')

    plt.tight_layout()

    if save2file:

        plt.savefig(filename)

        plt.close('all')

    else:

        plt.show()
im = generate_real(train_df, 10, (width, height))

print(im.dtype)

plot_image(im, False, dpi=10)
generated_images = generator.predict(np.random.normal(size=(batch_size, latent_dim)))

plot_images(save2file=False, fake=True, samples=12, images=generated_images, dpi=60)
#Should be a higher number. For learning purposes I have setted this value with a low value.

iterations = 5000



start = 0

for step in range(iterations):

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size

    real_images = np.zeros((batch_size, width, height, channels), dtype=np.float64)

    

    cont = 0

    for k in range(start, stop):

        real_images[cont] = generate_real(train_df, k, (width, height));

        cont += 1

    

    labels_real = np.ones((batch_size, 1))

    labels_fake = np.ones((batch_size, 1))

    

    # Add random noise to the labels - important trick!

    labels_real += 0.05 * np.random.random(labels_real.shape)

    labels_fake += 0.05 * np.random.random(labels_fake.shape)



    # Train the discriminator

    d_loss1 = discriminator.train_on_batch(real_images, labels_real)

    d_loss2 = discriminator.train_on_batch(generated_images, -labels_fake)

    d_loss = 0.5 * (d_loss1 + d_loss2)



    # sample random points in the latent space

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))



    # Assemble labels that say "all real images"

    misleading_targets = np.ones((batch_size, 1))



    # Train the generator (via the gan model,

    # where the discriminator weights are frozen)

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    

    start += batch_size

    if start > len(train_df) - batch_size:

        start = 0



    if step % 100 == 0:

        print('discriminator loss at step %s:\t %s \t-- adversarial loss at step %s:\t %s' % (step, d_loss, step, a_loss))

        

    # Occasionally save / plot

    if step % 500 == 0:

        showimages = np.concatenate([real_images[:4], generated_images[:8]])

        plot_images(save2file=False, fake=True, samples=12, images=showimages, dpi=60)

for k in range(0, 10):

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    generated_images = generator.predict(random_latent_vectors)

    plot_images(save2file=False, fake=True, samples=16, images=generated_images[:16], dpi=120)