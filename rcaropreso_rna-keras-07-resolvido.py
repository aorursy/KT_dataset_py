#CÉLULA KE-LIB-01

import numpy as np

import keras as K

import tensorflow as tf

import pandas as pd

import os

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#CÉLULA KE-LIB-02

np.random.seed(1)

tf.set_random_seed(1)
#CÉLULA KE-LIB-03

from tensorflow.examples.tutorials.mnist import input_data



x_train = input_data.read_data_sets("mnist", one_hot=True).train.images

x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
#CÉLULA KE-LIB-05

#Montando o Discriminador





tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



discriminator = K.Sequential()

depth = 64

dropout = 0.4

# In: 28 x 28 x 1, depth = 1

# Out: 14 x 14 x 1, depth=64

input_shape = (28, 28, 1)

discriminator.add(K.layers.Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))

discriminator.add(K.layers.LeakyReLU(alpha=0.2))

discriminator.add(K.layers.Dropout(dropout))



discriminator.add(K.layers.Conv2D(depth*2, 5, strides=2, padding='same'))

discriminator.add(K.layers.LeakyReLU(alpha=0.2))



discriminator.add(K.layers.Dropout(dropout))



discriminator.add(K.layers.Conv2D(depth*4, 5, strides=2, padding='same'))

discriminator.add(K.layers.LeakyReLU(alpha=0.2))

discriminator.add(K.layers.Dropout(dropout))



discriminator.add(K.layers.Conv2D(depth*8, 5, strides=1, padding='same'))

discriminator.add(K.layers.LeakyReLU(alpha=0.2))

discriminator.add(K.layers.Dropout(dropout))



# Out: 1-dim probability

discriminator.add(K.layers.Flatten())

discriminator.add(K.layers.Dense(1, activation='sigmoid'))



discriminator.summary()
#CÉLULA KE-LIB-06

#Montando o Gerador

tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



generator = K.Sequential()

depth = 64+64+64+64

dim = 7

dropout = 0.4



# In: 100

# Out: dim x dim x depth

generator.add(K.layers.Dense(dim*dim*depth, input_shape=(100,)))

generator.add(K.layers.BatchNormalization(momentum=0.9))

generator.add(K.layers.ReLU())



generator.add(K.layers.Reshape((dim, dim, depth)))

generator.add(K.layers.Dropout(dropout))



# In: dim x dim x depth

# Out: 2*dim x 2*dim x depth/2



generator.add(K.layers.UpSampling2D())

generator.add(K.layers.Conv2DTranspose(int(depth/2), 5, padding='same'))

generator.add(K.layers.BatchNormalization(momentum=0.9))

generator.add(K.layers.ReLU())



generator.add(K.layers.UpSampling2D())

generator.add(K.layers.Conv2DTranspose(int(depth/4), 5, padding='same'))

generator.add(K.layers.BatchNormalization(momentum=0.9))

generator.add(K.layers.ReLU())



generator.add(K.layers.Conv2DTranspose(int(depth/8), 5, padding='same'))

generator.add(K.layers.BatchNormalization(momentum=0.9))

generator.add(K.layers.ReLU())



# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix

generator.add(K.layers.Conv2DTranspose(1, 5, padding='same', activation='sigmoid' ))



generator.summary()
#CÉLULA KE-LIB-07

#Modelo do Discriminador

theOptimizer = K.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)

modelDM = K.models.Sequential()

modelDM.add(discriminator)

modelDM.compile(loss='binary_crossentropy', optimizer=theOptimizer, metrics=['accuracy'])



#Modelo do Adversário (Adversarial Model)

theOptimizer = K.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

modelAM = K.models.Sequential()

modelAM.add(generator)

modelAM.add(discriminator)

modelAM.compile(loss='binary_crossentropy', optimizer=theOptimizer, metrics=['accuracy'])
#CÉLULA KE-LIB-08

def plot_images(save2file=False, fake=True, samples=16, noise=None, step=0):

        filename = 'mnist.png'

        if fake:

            if noise is None:

                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])

            else:

                filename = "mnist_%d.png" % step

            images = generator.predict(noise)

        else:

            i = np.random.randint(0, x_train.shape[0], samples)

            images = x_train[i, :, :, :]



        plt.figure(figsize=(10,10))

        for i in range(images.shape[0]):

            plt.subplot(4, 4, i+1)

            image = images[i, :, :, :]

            image = np.reshape(image, [28, 28])

            plt.imshow(image, cmap='gray')

            plt.axis('off')

        plt.tight_layout()

        if save2file:

            plt.savefig(filename)

            plt.close('all')

        else:

            plt.show()
#CÉLULA KE-LIB-09

#Treinamento

max_epochs = 2000

batch_size = 256



noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])



# for i in range(max_epochs):

#     images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]

#     noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

#     images_fake = generator.predict(noise)

#     x = np.concatenate((images_train, images_fake))

#     y = np.ones([2*batch_size, 1])

#     y[batch_size:, :] = 0



#     d_loss = modelDM.train_on_batch(x, y)

#     y = np.ones([batch_size, 1])

#     noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])



#     a_loss = modelAM.train_on_batch(noise, y)



#     log_mesg = print("%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1]))

#     log_mesg = print("%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1]))

#     plot_images(save2file=False, samples=noise_input.shape[0], noise=noise_input, step=(i+1))