import numpy as np

import pandas as pd

import cv2

from matplotlib import pyplot as plt



import time

import os
IMG_SIZE = 64

SAMPLE_COUNT = 200

TRAIN_DATA = '/kaggle/input/manwomandetection/dataset/dataset/train'



NUM_CLASSES = 2

# grayscale only

CHANNELS = 1



data = np.zeros([SAMPLE_COUNT, IMG_SIZE, IMG_SIZE, CHANNELS])

labels = np.zeros([SAMPLE_COUNT])

i = 0

for dirname, _, filenames in os.walk(TRAIN_DATA):

    # obtain class name from the directories name

    class_name = dirname.split('/')[-1]

    # read all files inside the directory

    for filename in filenames:        

        # read image

        img = cv2.imread(os.path.join(dirname, filename), cv2.IMREAD_GRAYSCALE)        

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        

        

        # normalize to -1.0 to 1.0 range

        img = img / 127.5 - 1.0

        

        # scale so that one dimension is equal to IMG_SIZE and the other is smaller

        # preserve image ratio

        scale = IMG_SIZE / max(img.shape[:2])

        img = cv2.resize(img, (0,0), fx=scale, fy=scale)

        

        # get image dimensions

        h, w = img.shape[:2]

        if w > h:

            # calculate vertical offset

            offset = (w - h) // 2

            # add image to dataset

            data[i,offset:h+offset,:,0] = img

        else:

            # calculate horizontal offset

            offset = (h - w) // 2

            # add image to dataset

            data[i,:,offset:w+offset,0] = img

        

        # set label

        if class_name == 'man':

            labels[i] = 0

        elif class_name == 'woman':

            labels[i] = 1

        else:

            print('wait what?')

        

        i += 1

        

        # load half of the samples from one directory and the other half from the other dir

        if i % (SAMPLE_COUNT // 2) == 0:

            break

# show some examples

EXAMPLES = 5

# figure has five subplots next to each other

fig, axs = plt.subplots(1, EXAMPLES, figsize=(15,8))

for i in range(EXAMPLES):

    # pick a random sample

    r = np.random.randint(0, data.shape[0])

    # show the grayscale image

    axs[i].imshow(data[r,:,:,0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')

    # print the corresponding class

    print(labels[r])



plt.show()
from keras.layers import Dense, Flatten, Reshape, Embedding, Input

from keras.layers import BatchNormalization, Multiply, Dropout, Concatenate

from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.utils import to_categorical



LATENT_SIZE = 100



class CGAN():

    def __init__(self):

        self.optimizer = RMSprop(lr=0.0001)

        

        self.generator = self.build_generator()        

        

        self.frozen_generator = Model(inputs=self.generator.inputs, outputs=self.generator.outputs)

        self.frozen_generator.trainable = False        

        

        self.discriminator = self.build_discriminator()

        

        real_image = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

        noise = Input(shape=(LATENT_SIZE,))



        label = Input(shape=(1,))

        gen_label = Input(shape=(1,))

        

        gen_img = self.frozen_generator([noise, gen_label])

        fake = self.discriminator([gen_img, gen_label])

        

        valid = self.discriminator([real_image, label])

        

        self.critic = Model(inputs=[real_image, label, noise, gen_label], outputs=[fake, valid])

        self.critic.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=self.optimizer)

        self.critic.summary()

        

        self.frozen_discriminator = Model(inputs=self.discriminator.inputs, outputs=self.discriminator.outputs)

        self.frozen_discriminator.trainable = False

        

        fake_img = self.generator([noise, label])



        validity = self.frozen_discriminator([fake_img, label])

        

        self.combined = Model(inputs=[noise, label], outputs=validity)

        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.combined.summary()

        

        

    def build_generator(self):

        # define the standalone discriminator model

        in_label = Input(shape=(1,))

        in_latent = Input(shape=(LATENT_SIZE,))                

        

        x = Multiply()([Embedding(NUM_CLASSES, LATENT_SIZE)(in_label), in_latent])

        

        x = Dense(400)(x)

        x = LeakyReLU(alpha=.2)(x)

        x = BatchNormalization(momentum=.8)(x)

        

        x = Dense(1000)(x)

        x = LeakyReLU(alpha=.2)(x)

        x = BatchNormalization(momentum=.8)(x)

        

        x = Dropout(.4)(x)        

        

        x = Dense(IMG_SIZE * IMG_SIZE * CHANNELS, activation='tanh')(x)



        out = Reshape([IMG_SIZE, IMG_SIZE, CHANNELS])(x)

        

        return Model(inputs=[in_latent, in_label], outputs=out)

        

    def build_discriminator(self):

        in_label = Input(shape=(1,))

        in_image = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

        

        x1 = Embedding(NUM_CLASSES, 50)(in_label)

        x1 = Dense(IMG_SIZE * IMG_SIZE * CHANNELS)(x1)

        x1 = LeakyReLU(alpha=.2)(x1)

        # reshape to image dimensions

        x1 = Reshape((IMG_SIZE, IMG_SIZE, CHANNELS))(x1)



        x = Concatenate()([x1, in_image])

        

        x = Conv2D(256, (3,3), strides=(2,2), padding='same')(x)

        x = LeakyReLU(alpha=.2)(x)



        x = Conv2D(256, (3,3), strides=(2,2), padding='same')(x)

        x = LeakyReLU(alpha=.2)(x)

        x = Flatten()(x)    

        

        x = Dropout(.4)(x)

        x = Dense(100)(x)

        x = LeakyReLU(alpha=.2)(x)

        x = Dropout(.4)(x)



        out = Dense(1, activation='sigmoid')(x)

        

        return Model(inputs=[in_image, in_label], outputs=out)

    

    def save_images(self, epoch):

        images = gan.generator.predict([np.random.normal(0, 1, [2, LATENT_SIZE]), np.asarray([0, 1])])

        for i in range(2):

            name = ('man' if i == 0 else 'woman') + ('_epoch_%4d' % epoch) + '.png'

            cv2.imwrite(name, images[i])

            

    def show_images(self):

        images = self.generator.predict([np.random.normal(0, 1, [2, LATENT_SIZE]), np.asarray([0, 1])])



        fig, axs = plt.subplots(1, 2, figsize=(10,6))

        for i in range(2):

            axs[i].imshow(images[i,:,:,0], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')



        plt.show()





    def train(self, epochs=100, batch_size=25):

        # loop over epochs

        for epoch in range(epochs+1):

            epoch_fake_loss = 0

            epoch_real_loss = 0

            epoch_gen_loss = 0

            

            # shuffled array of indizes

            idx = np.arange(0, data.shape[0])

            np.random.shuffle(idx)

            

            cnt = 0

            start = time.time()

            # train on each batch

            for i in range(0, data.shape[0], batch_size):

                

                size = batch_size

                # the last batch may have a different size                

                if i + batch_size > data.shape[0]:

                    # adjust size so there is no overflow

                    size = data.shape[0] - i

                

                # generate noise and label vector

                noise = np.random.normal(0, 1, [size, LATENT_SIZE])

                fake_labels = np.random.randint(0, NUM_CLASSES, size)

                

                # real samples and their corresponding labels

                real_labels = labels[idx[i:i+size]]

                real_images = data[idx[i:i+size]]

                

                # just arrays of ones or zeros

                valid = np.ones([size])

                fake = np.zeros([size])

                

                # train discriminator

                fake_loss, real_loss = self.critic.train_on_batch([real_images, real_labels, noise, fake_labels], [fake, valid])[1:]                

                # train generator

                gen_loss = self.combined.train_on_batch([noise, fake_labels], valid)

                

                # sum losses

                epoch_fake_loss += fake_loss

                epoch_real_loss += real_loss

                epoch_gen_loss += gen_loss

                

                cnt += 1

            

            # compute the epochs loss average

            epoch_fake_loss /= cnt

            epoch_real_loss /= cnt

            epoch_gen_loss /= cnt

            

            # 

            duration = time.time() - start

            # print epoch number, losses and the duration

            print('#%d %.2f seconds [D fake=%.4f real=%.4f - G %.4f]' % (epoch, duration, epoch_fake_loss, epoch_real_loss, epoch_gen_loss))

            

            # save samples (one generated male and female image) every ten epochs

            if epoch % 10 == 0 and epoch != 0:

                #self.save_images(epoch)

                self.show_images()          

            

            

    def train_generator_only(self, epochs=100, batch_size=25):

        valid = np.ones([batch_size])

        

        for epoch in range(epochs+1):

            noise = np.random.normal(0, 1, [batch_size, LATENT_SIZE])

            fake_labels = np.random.randint(0, NUM_CLASSES, batch_size)



            start = time.time()

            gen_loss = self.combined.train_on_batch([noise, fake_labels], valid)

            duration = time.time() - start

            

            print('#%d %.2f seconds [G %.4f]' % (epoch, duration, gen_loss))

            

            if epoch % 10 == 0 and epoch != 0:

                #self.save_images(epoch)

                self.show_images() 

            

                



gan = CGAN()
gan.train(epochs=1000)

gan.show_images()