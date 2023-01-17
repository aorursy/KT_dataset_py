# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout

from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import UpSampling2D, Conv2D

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import initializers

import matplotlib.pyplot as plt

import sys

import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Set the seed

np.random.seed(100)



random_Dim = 10



# Load MNIST Data

(X_train, X_test), (y_train, y_test) = mnist.load_data()



# Making pixel range fall in between [-1, 1]

X_train = (X_train.astype(np.float32)-127.5)/127.5

#X_train = X_train.reshape(60000, 784)

X_train.shape
# Reshape X_train 

X_train = X_train.reshape(60000, 784)
adam = Adam(lr=0.0002, beta_1=0.5)



# Create Generator Sequence

generator = Sequential()

generator.add(Dense(256, input_dim=random_Dim))

generator.add(LeakyReLU(0.2))

generator.add(Dense(512))

generator.add(LeakyReLU(0.2))

generator.add(Dense(1024))

generator.add(LeakyReLU(0.2))

generator.add(Dense(784, activation='tanh'))
discriminator = Sequential()

discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Dropout(0.3))

discriminator.add(Dense(512))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Dropout(0.3))

discriminator.add(Dense(256))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Dropout(0.3))

discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=adam)
# Combined GAN network

discriminator.trainable=False

ganInput = Input(shape=(random_Dim,))

x= generator(ganInput)

ganOutput = discriminator(x)

gan = Model(inputs=ganInput, outputs=ganOutput)

gan.compile(loss='binary_crossentropy', optimizer=adam)



dLosses = []

gLosses = []
# Let's plot the losses from epoch



def plotLoss(epoch):

    plt.figure(figsize=(10, 8))

    plt.plot(dLosses, label='Discriminitive loss')

    plt.plot(gLosses, label='Generative loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.savefig('/kaggle/working/gan_loss_epoch_%d.png' % epoch)

    
# Create Generated MNIST images



def saveGeneratedImages(epoch, examples=100, dim=(10,10), figsize=(10,10)):

    noise = np.random.normal(0,1, size=[examples, random_Dim])

    generatedImages = generator.predict(noise)

    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)

    

    for i in range(generatedImages.shape[0]):

        plt.subplot(dim[0], dim[1], i+1)

        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')

        plt.axis('off')

    plt.tight_layout()

    plt.savefig('/kaggle/working/gan_generated_image_epoch_%d.png' % epoch)
def train(epochs=1, batchSize=128):

    batchCount = int(X_train.shape[0]/batchSize)

    print ('Epochs:', epochs)

    print ('Batch size:', batchSize)

    print ('Batches per epoch:', batchCount)

    

    for epoch in range(1, epochs+1):

        print ('-'*15, 'Epoch %d' % epoch, '-'*15)

        for _ in range(batchCount):

            # Get noise and their images

            noise = np.random.normal(0,1, size=[batchSize, random_Dim])

            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            

            # Generate fake MNIST images

            generatedImages = generator.predict(noise)

            

            # print shapes

            X = np.concatenate([imageBatch, generatedImages])

            

            # Labels for generated and real images

            yDis = np.zeros(2*batchSize)

            

            # Smooth label

            yDis[:batchSize] = 0.9

            

            # Train Discriminator

            discriminator.trainable = True

            dloss = discriminator.train_on_batch(X, yDis)

            

            # Train Generator

            noise = np.random.normal(0,1, size=[batchSize, random_Dim])

            yGen = np.ones(batchSize)

            discriminator.trainable=False

            gloss = gan.train_on_batch(noise, yGen)

            

        dLosses.append(dloss)

        gLosses.append(gloss)

        

        if epoch==1 or epoch%20 == 0:

            saveGeneratedImages(epoch)

            

    plotLoss(epoch)

            
train(200, 128)
!ls /kaggle/working