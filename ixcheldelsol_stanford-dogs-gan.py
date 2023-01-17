# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function, division 

import time 

 

import tarfile



import sys



import numpy as np 



import matplotlib.pyplot as plt 

import matplotlib.image as mpimg 



from PIL import Image



import cv2



import pandas as pd



from keras.models import Model, Sequential  

from keras.layers import Input, Dense, Conv2D, Activation, ZeroPadding2D, UpSampling2D, Reshape, BatchNormalization, Flatten, Dropout  

from keras.layers.advanced_activations import LeakyReLU 

from keras.preprocessing.image import ImageDataGenerator 

from keras.optimizers import Adam 
ANNOTATION_DIR = '../input/annotations/Annotation/'

IMAGES_DIR = '../input/images/Images/'



#list of breeds of dogs in the dataset

breed_list = os.listdir(ANNOTATION_DIR)
class GAN ():

    def _init_(self): 

        self.img_rows = 28

        self.img_cols = 28 

        self.channels = 3 

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = 100

        

        optimizer = Adam(0.002, 0.5)

        

        #Build and compile the discriminator 

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        

        #Build the generator

        self.generator = self.build_generator()

        

        #The generator takes noise as input and generates images 

        z = Input(shape = (self.latent_dim))

        img = self.generator(z)

        

        # For the combined model we will only train the generator 

        self.discriminator.trainable = False

        

        #The discriminator takes generated images as input and determines validity 

        validity = self.discriminator(img)

        

        #Combined model (stacked generator and discriminator)

        #Train generator to fool the discriminator 

        self.combined = Model(z, validity)

        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        

    def build_generator(self): 

        

        model = Sequential()

        

        model.add(Dense(256, input_dim = self.latent_dim))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.image_shape), activation='tanh'))

        model.add(Reshape(self.img_shape))

        

        model.summary()

 

        noise = Input(shape=(self.latent_dim,))

        img = model(noise)

        

        return Model(noise, img)

    

    def build_discriminator(self): 

        

        model = Sequential()

        

        model.add(Flatten(input_shape=self.img_shape))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation = 'sigmoid'))

        

        model.summary()

        

        img = Input(shape=self.img_shape)

        validity = model(img)

        

        return Model(img, validity)

    

    #Entrenamiento de la GAN 

    

    def train(self, epochs, batch_size=128, sample_interval=50): 

        

        #Load the dataset --> Hay que hacerlo con los Stanford Dogs porque solo consegui MNIST

        

        

        # targets 

        valid = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        

        for epoch in range(epochs):



            # ---------------------

            #  Train Discriminator

            # ---------------------



            # Select a random batch of images

            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs = X_train[idx]



            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))



            # Generate a batch of new images

            gen_imgs = self.generator.predict(noise)



            # Train the discriminator

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            # ---------------------

            #  Train Generator

            # ---------------------



            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))



            # Train the generator (to have the discriminator label samples as valid)

            g_loss = self.combined.train_on_batch(noise, valid)



            # Plot the progress

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



            # If at save interval => save generated image samples

            if epoch % sample_interval == 0:

                self.sample_images(epoch)

                

        def sample_images(self, epoch): #Impresion de las imagenes 

            r, c = 5, 5

            noise = np.random.normal(0, 1, (r * c, self.latent_dim))

            gen_imgs = self.generator.predict(noise)



            # Rescale images 0 - 1

            gen_imgs = 0.5 * gen_imgs + 0.5



            fig, axs = plt.subplots(r, c)

            cnt = 0

            for i in range(r):

                for j in range(c):

                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')

                    axs[i,j].axis('off')

                    cnt += 1

            fig.savefig("images/%d.png" % epoch)

            plt.close()

    

    

   



        

        
 #Creacion del modelo y corrida de la GAN 

    if __name__ == '__main__':

        gan = GAN()

        gan.train(epochs=30000, batch_size=32, sample_interval=200)