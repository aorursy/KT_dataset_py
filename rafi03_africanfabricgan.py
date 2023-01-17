# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(filenames)
from PIL import Image

from numpy import asarray

import numpy as np

from numpy import zeros

from numpy import ones

from numpy.random import randn

from numpy.random import randint

from keras.optimizers import Adam

from keras.models import Sequential

from matplotlib import pyplot

from keras.layers import BatchNormalization

from keras.initializers import RandomNormal

from keras.utils.vis_utils import plot_model

from keras.models import Model

from keras.layers import (Input, MaxPooling2D, GlobalAveragePooling2D, 

                          UpSampling2D,Conv2DTranspose,LeakyReLU,Dropout,

                          Activation,Dense,Flatten,Reshape,Conv2D)
dataset_orig = np.empty((len(filenames), 64, 64,3))
ind=0

for i in filenames:

    image = Image.open(os.path.join(dirname, i))

    data = asarray(image)

    dataset_orig[ind]=data

    ind+=1
dataset_orig.shape
width, height, channel = 64, 64, 3

np.random.shuffle(dataset_orig)

X=dataset_orig
X = (X - 127.5) / 127.5
X.shape
import matplotlib.pyplot as plt

def show_data(X, title=""):

    plt.figure(figsize=(11,11))

    

    i = 1

    for img in X:

        plt.subplot(10, 10, i)

        plt.imshow(img.reshape((height, width,channel)))

        plt.axis('off')

        i+=1

        if i>100: break



    plt.suptitle(title, fontsize = 25)

    plt.show()

    

show_data(X, title="Original Dataset")
gen_optimizer = Adam(0.0001, 0.5)

disc_optimizer = Adam(0.0002, 0.5)

noise_dim = 100
def buildGenerator():

    model = Sequential()



    model.add(Dense(1024, input_dim=noise_dim))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    

    model.add(Dense(8192, input_dim=noise_dim))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    

    model.add(Reshape((8, 8, 128)))

    

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(64, (2, 2), padding='same', 

                     kernel_initializer=RandomNormal(0, 0.02)))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(64, (2, 2), padding='same', 

                     kernel_initializer=RandomNormal(0, 0.02)))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(channel, (3, 3), padding='same', activation = "tanh", 

                     kernel_initializer=RandomNormal(0, 0.02)))

    

    return model
generator = buildGenerator()

generator.summary()

# plot the model

plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
def buildDiscriminator():

    model = Sequential()

    

    model.add(Conv2D(64, (5, 5), strides=2, padding='same', 

                     kernel_initializer=RandomNormal(0, 0.02), 

                     input_shape=(width, height, channel)))

    model.add(LeakyReLU(0.2))





    model.add(Conv2D(64, (5, 5), strides=2, padding='same', 

                     kernel_initializer=RandomNormal(0, 0.02), 

                     input_shape=(width, height, channel)))

    model.add(LeakyReLU(0.2))

    

    model.add(Conv2D(128, (5, 5), strides=2,padding='same', 

                     kernel_initializer=RandomNormal(0, 0.02)))

    model.add(LeakyReLU(0.2))

    

    model.add(Flatten())

    

    model.add(Dense(256))

    model.add(LeakyReLU(0.2))

    

    model.add(Dropout(0.5))



    

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(loss='binary_crossentropy', optimizer=disc_optimizer)

    return model
discriminator = buildDiscriminator()

discriminator.summary()

# plot the model

plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
noise = Input(shape=(noise_dim,))

fake_data = generator(noise)

discriminator.trainable = False

output = discriminator(fake_data)

gan = Model(noise, output)

gan.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
gan.summary()

# plot the model

plot_model(gan, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
fixed_noise = np.random.normal(0, 1, size=(100, noise_dim))
def show_generated_fabric(title, epoch):

    imgs = generator.predict(fixed_noise)

    imgs = 0.5 * imgs + 0.5

    plt.figure(figsize=(11,11))

    

    i = 1

    for img in imgs:

        plt.subplot(10, 10, i)

        plt.imshow(img.reshape((height,width,channel)))

        plt.axis('off')

        i+=1

    plt.suptitle(title, fontsize = 25)

    plt.savefig(str(epoch+1)+".png", transparent=True)

    plt.show()
epochs = 500

batch_size = 128

steps_per_epoch = len(X)//batch_size
for epoch in range(epochs):

    for batch in range(steps_per_epoch):

        input_gen = np.random.normal(0, 1, size=(batch_size, noise_dim))

        fake_data = generator.predict(input_gen)

        

        real_data = X[np.random.randint(0, X.shape[0], size=batch_size)]

        real_data = real_data.reshape((batch_size, width, height, channel))

        

        input_disc = np.concatenate((real_data, fake_data))



        label_disc = np.zeros(2*batch_size)

        label_disc[:batch_size] = 0.9

        label_disc[batch_size:] = 0.1

        loss_disc = discriminator.train_on_batch(input_disc, label_disc)



        label_gen = np.ones(batch_size)

        loss_gen = gan.train_on_batch(input_gen, label_gen)



    print("epoch: ", epoch)

    print("discriminator loss: ", loss_disc)

    print("generator loss: ", loss_gen)

    print("-"*80)

    

    if (epoch+1) % 20 == 0:

        show_generated_fabric("Generated Fabric", epoch)

        filename = 'generator_model_%03d.h5' % (epoch+1)

        generator.save(filename)
!zip generated_pics.zip *.png
!zip models.zip *.h5
from keras.models import load_model

model = load_model('generator_model_440.h5')
fixed_noise = np.random.normal(0, 1, size=(100, noise_dim))

imgs = model.predict(fixed_noise)

imgs = 0.5 * imgs + 0.5

plt.figure(figsize=(11,11))

i = 1

for img in imgs:

    plt.subplot(10, 10, i)

    plt.imshow(img.reshape((height,width,channel)))

    plt.axis('off')

    i+=1

plt.show()
fixed_noise = np.random.normal(0, 1, size=(100, noise_dim))

imgs = model.predict(fixed_noise)

imgs = 0.5 * imgs + 0.5

plt.figure(figsize=(11,11))

i = 1

for img in imgs:

    plt.subplot(10, 10, i)

    plt.imshow(img.reshape((height,width,channel)))

    plt.axis('off')

    i+=1

plt.show()
# define the standalone discriminator model

def define_discriminator(in_shape=(64,64,3)):

    # weight initialization

    init = RandomNormal(0, 0.02)



    model = Sequential()

    # normal

    model.add(Conv2D(64, (5,5), padding='same', input_shape=in_shape))

    #model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))

    #model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))

    #model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))

    #model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))

    #model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # classifier

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

# define model

model = define_discriminator()

# summarize the model

model.summary()

# plot the model

plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)



# define the standalone generator model

def define_generator(latent_dim):

    # weight initialization

    #init = RandomNormal(stddev=0.02)

    model = Sequential()

    # foundation for 4x4 image

    n_nodes = 4*4*128

    model.add(Dense(n_nodes, input_dim=latent_dim))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((4, 4, 128)))

    # upsample to 8x8

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # upsample to 16x16

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # upsample to 32x32

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # upsample to 64x64

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    # output layer

    model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))

    return model

# define the size of the latent space

latent_dim = 100

# define the generator model

model = define_generator(latent_dim)

# summarize the model

model.summary()

# plot the model

plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

# define the combined generator and discriminator model, for updating the generator

def define_gan(g_model, d_model):

    # make weights in the discriminator not trainable

    d_model.trainable = False

    # connect them

    model = Sequential()

    # add generator

    model.add(g_model)

    # add the discriminator

    model.add(d_model)

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model

latent_dim = 100

# create the discriminator

d_model = define_discriminator()

# create the generator

g_model = define_generator(latent_dim)

# create the gan

gan_model = define_gan(g_model, d_model)

# summarize gan model

gan_model.summary()

# plot gan model

plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)