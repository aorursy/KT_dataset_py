from __future__ import print_function, division



from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import Adam



import matplotlib.pyplot as plt



import sys



import numpy as np

from torchvision.utils import save_image
class GAN():

    def __init__(self):

        self.img_rows = 28

        self.img_cols = 28

        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = 100   # noise input length

        

        optimizer = Adam(0.0002, 0.5) # what is beta1?

        

        # build and compile discriminator

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        

        # build generator

        self.generator = self.build_generator()

#         self.generator.compile(loss = 'binary_crossentropy', optimizer = optimizer)  # why no metrics????????

        

        # generator takes noise as input, and images as output

        z = Input(shape=(self.latent_dim,))

        img = self.generator(z)

        

        # in combined model, only train generator

        # why is that?

        self.discriminator.trainable = False 

        

        # discriminator takes generated image and deicde whether it is generated

        judge = self.discriminator(img)

        

        # the combined model (stack generator and discriminator)

        self.combined = Model(z, judge)

        self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer)

        

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim = self.latent_dim))

        model.add(LeakyReLU(alpha = 0.2))

        model.add(BatchNormalization(momentum = 0.8))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha = 0.2))

        model.add(BatchNormalization(momentum = 0.8))

        model.add(Dense(1024))

        model.add(LeakyReLU(alpha = 0.2))

        model.add(BatchNormalization(momentum = 0.8))

        model.add(Dense(np.prod(self.img_shape), activation = 'tanh'))

        model.add(Reshape(self.img_shape))

        model.summary()



        noise = Input( shape = (self.latent_dim, ))

        img = model(noise)



        return Model(noise, img)



    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape = self.img_shape))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha = 0.2))

        model.add(Dense(256))

        model.add(LeakyReLU(alpha = 0.2))

        model.add(Dense(1, activation = 'sigmoid'))



        model.summary()



        img = Input(shape = self.img_shape)

        judge = model(img)



        return Model(img, judge)





    def train(self, epochs, batch_size = 128, save_interval = 50):

        # load data

        (X_train, _), (_,_) = mnist.load_data()    # need internet

        

        # rescale -1 to 1

        X_train = X_train / 127.5 - 1

        X_train = np.expand_dims(X_train, axis = 3)



        # adversarial ground truths

        valid = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))



        for epoch in range(epochs):

            # -----------

            # Train discriminator

            # -----------



            # select a batch of real images

            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs = X_train[idx]



            # generate a batch of images

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_imgs = self.generator.predict(noise)



            # train discriminator

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            # -----------

            # Train generator

            # -----------



            noise = np.random.normal(0, 1, (batch_size,self.latent_dim ))



            g_loss = self.combined.train_on_batch(noise, valid)



            # plot progress

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % \

                 (epoch, d_loss[0], 100*d_loss[1], g_loss))



            # if at save interval -> save generated image samples

            if epoch % save_interval == 0:

                self.save_image(epoch)





    def save_image(self, epoch):

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        gen_imgs = self.generator.predict(noise)



        # rescale image 0 - 1

        gen_imgs = 0.5 * gen_imgs + 0.5



        fig, axs = plt.subplots(r, c)

        cnt = 0 #??

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap = 'gray')

                axs[i,j].axis('off')

                cnt += 1

        fig.savefig("../output_images/%d.png" % epoch)

#         save_image(fig, os.path.join(

#             '../output_images', '%d.png' % epoch))

        plt.close()
import shutil

import os 

! mkdir ../output_images

if __name__ == '__main__':

    gan = GAN()

    gan.train(epochs = 5000, batch_size = 32, save_interval = 200)

    shutil.make_archive('images', 'zip', '../output_images')
(X_train, _), (_,_) = mnist.load_data()    # need internet
(X_train, _), (_,_) = mnist.load_data()    # need internet



# rescale -1 to 1

X_train = X_train / 127.5 - 1

X_train = np.expand_dims(X_train, axis = 3)
X_train.shape
(X_train, _), (_,_) = mnist.load_data()    # need internet

X_train.shape