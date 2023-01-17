import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from PIL import Image

from matplotlib import pyplot as plt



WIDTH = 128

HEIGHT = 128



from keras import Input

from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout

from keras.models import Model

from keras.optimizers import RMSprop



LATENT_DIM = 32

CHANNELS = 3



def create_generator():

    gen_input = Input(shape=(LATENT_DIM, ))

    

    x = Dense(128 * 16 * 16)(gen_input)

    x = LeakyReLU()(x)

    x = Reshape((16, 16, 128))(x)

    

    x = Conv2D(256, 5, padding='same')(x)

    x = LeakyReLU()(x)

    

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)

    x = LeakyReLU()(x)

    

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)

    x = LeakyReLU()(x)

    

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(512, 5, padding='same')(x)

    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)

    x = LeakyReLU()(x)

    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    

    generator = Model(gen_input, x)

    return generator



def create_discriminator():

    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    

    x = Conv2D(256, 3)(disc_input)

    x = LeakyReLU()(x)

    

    x = Conv2D(256, 4, strides=2)(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(256, 4, strides=2)(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(256, 4, strides=2)(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(256, 4, strides=2)(x)

    x = LeakyReLU()(x)

    

    x = Flatten()(x)

    x = Dropout(0.4)(x)

    

    x = Dense(1, activation='sigmoid')(x)

    discriminator = Model(disc_input, x)

    

    optimizer = RMSprop(

        lr=.0001,

        clipvalue=1.0,

        decay=1e-8

    )

    

    discriminator.compile(

        optimizer=optimizer,

        loss='binary_crossentropy'

    )

    

    return discriminator



generator = create_generator()

discriminator = create_discriminator()

discriminator.trainable = False



gan_input = Input(shape=(LATENT_DIM, ))

gan_output = discriminator(generator(gan_input))

gan = Model(gan_input, gan_output)



optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)

gan.compile(optimizer=optimizer, loss='binary_crossentropy')



# load pretrained weights (see part 1 of this kernel)

gan.load_weights('/kaggle/input/faces_gan.h5')
EXPLORE_FEATURES = 32

SHIFTS = [-2, -1, -.5, -.2, -.1, 0, .1, .2, .5, 1, 2]



# visualize possible tunings for defining features meaning in latent vector

plt.figure(1, figsize=(2 * len(SHIFTS), EXPLORE_FEATURES * 2))

i = 0

for f_pos in range(EXPLORE_FEATURES):

    for shift in SHIFTS:

        i += 1

        plt.subplot(EXPLORE_FEATURES, len(SHIFTS), i)

        latent_vector = np.zeros((1, LATENT_DIM))

        latent_vector[0, f_pos] = shift

        img = generator.predict(latent_vector)[0]

        plt.imshow(img)

        plt.axis('off')

plt.show()
# we can define user-friendly features

def create_latent_vector(smile=0, thickness=0, male=0, light_hair=0, short_hair=0):

    vec = np.zeros((1, LATENT_DIM))

    

    # forw is a list of features with accelerating effect

    # back is a list of features with descelerating effect

    def fill_vec(forw, back, coef):

        vec = np.zeros((LATENT_DIM, ))

        for pos in forw:

            vec[pos] = 1

        for pos in back:

            vec[pos] = -1

        vec *= coef

        return vec

    

    vec[0] += fill_vec([1, 3, 6], [7, 9, 14, 23, 28, 29, 30], smile)

    vec[0] += fill_vec([5, 8, 10, 15, 20], [9, 22, 23, 24], thickness)

    vec[0] += fill_vec([2, 5, 17, 30], [0, 1, 12, 18, 25, 28], male)

    vec[0] += fill_vec([2, 12], [13, 31], light_hair)

    vec[0] += fill_vec([2, 5], [0, 6, 18, 20], short_hair)

    

    # we can see that some feature may correlate

    

    return vec



def plot_images(vectors):

    plt.figure(1, figsize=(18, 2))

    i = 0

    for vec in vectors:

        i += 1

        plt.subplot(1, 9, i)

        img = generator.predict(vec)[0]

        plt.imshow(img)

        plt.axis('off')

    plt.show()



SHIFTS = [-1, -.7, -.4, -.2, 0, .2, .4, .7, 1]



# over smiles

plot_images([create_latent_vector(smile=shift) for shift in SHIFTS])
# over gender

plot_images([create_latent_vector(male=shift) for shift in SHIFTS])
# over thickness

plot_images([create_latent_vector(thickness=shift) for shift in SHIFTS])
plt.figure(1, figsize=(6, 6))

# man without smile with light hair

img = generator.predict(create_latent_vector(male=1, smile=-1, light_hair=1))[0]

plt.imshow(img)

plt.axis('off')

plt.show()
plt.figure(1, figsize=(6, 6))

# man with long hair

img = generator.predict(create_latent_vector(male=1, short_hair=-1))[0]

plt.imshow(img)

plt.axis('off')

plt.show()
plt.figure(1, figsize=(6, 6))

# thick woman

img = generator.predict(create_latent_vector(male=-1, thickness=1))[0]

plt.imshow(img)

plt.axis('off')

plt.show()
plt.figure(1, figsize=(6, 6))

# thin man

img = generator.predict(create_latent_vector(male=1, thickness=-1))[0]

plt.imshow(img)

plt.axis('off')

plt.show()