 


from __future__ import print_function, division

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import pandas as pd

import numpy as np

import os

from tqdm import tqdm, tqdm_notebook

from keras.preprocessing.image import load_img

from keras.applications.densenet import preprocess_input

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import array_to_img

from keras.preprocessing import image

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.







img_size = 64

batch_size = 128







def resize_to_square(im):

    # new_size should be in (width, height) format

    im = cv2.resize(im, (img_size, img_size))

    return im



def load_image2(file):

    image = cv2.imread(file)

    new_image = resize_to_square(image)

    #ew_image = preprocess_input(new_image)

    return new_image



def load_image(file):

    new_image = load_img(file, target_size=(img_size, img_size))

    new_image = (img_to_array(new_image))

    #new_image = resize_to_square(img_to_array(new_image))

    #ew_image = preprocess_input(new_image)

    return new_image

    

if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

    





from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise

from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D

from keras.layers import MaxPooling2D, concatenate

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras import losses

from keras.utils import to_categorical

import keras.backend as K



import matplotlib.pyplot as plt



import numpy as np







from keras.preprocessing.image import ImageDataGenerator
all_file=os.listdir("../input/all-dogs/all-dogs/")

train_datagen = ImageDataGenerator()



train_generator = train_datagen.flow_from_directory(

        '../input/all-dogs/',

        target_size=(64, 64),

        batch_size=batch_size)



train_datagen_augment = ImageDataGenerator( featurewise_center=True,

        featurewise_std_normalization=True,

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        horizontal_flip=True,

        shear_range=0.2)





train_generator_augment = train_datagen_augment.flow_from_directory(

        '../input/all-dogs/',

        target_size=(64, 64),

        batch_size=batch_size)
plt.imshow(image.array_to_img(train_generator[1][0][0]))

plt.show()



plt.imshow(image.array_to_img(train_generator_augment[1][0][0]))

plt.show()
# Load the dataset

#all_file=os.listdir("../input/all-dogs/all-dogs/")



#x_train_data = np.zeros((len(all_file),img_size,img_size,3))

#for i in range(len(all_file)-1):

#    file=all_file[i]

#    path="../input/all-dogs/all-dogs/"+file

#    x_train_data[i] = load_image2(path)


class BIGAN():

    def __init__(self):

        self.img_rows = 64

        self.img_cols = 64

        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.img_shape2 = (64, 64, self.channels)

        self.latent_dim = 100



        optimizer = Adam(0.001, 0.5)



        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss=['binary_crossentropy'],

            optimizer=optimizer,

            metrics=['accuracy'])



        # Build the generator

        self.generator = self.build_generator()



        # Build the encoder

        self.encoder = self.build_encoder()



        # The part of the bigan that trains the discriminator and encoder

        self.discriminator.trainable = False



        # Generate image from sampled noise

        z = Input(shape=(self.latent_dim, ))

        img_ = self.generator(z)



        # Encode image

        img = Input(shape=self.img_shape)

        z_ = self.encoder(img)



        # Latent -> img is fake, and img -> latent is valid

        fake = self.discriminator([z, img_])

        valid = self.discriminator([z_, img])



        # Set up and compile the combined model

        # Trains generator to fool the discriminator

        self.bigan_generator = Model([z, img], [fake, valid])

        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],

            optimizer=optimizer)





    def build_encoder(self):

        model = Sequential()



        model.add(Flatten(input_shape=self.img_shape))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.latent_dim))



        model.summary()



        img = Input(shape=self.img_shape)

        z = model(img)



        return Model(img, z)



    def build_generator(self):

        model = Sequential()



        model.add(Dense(512, input_dim=self.latent_dim))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))

        model.add(Reshape(self.img_shape))



        model.summary()



        z = Input(shape=(self.latent_dim,))

        gen_img = model(z)



        return Model(z, gen_img)



    def build_discriminator(self):



        z = Input(shape=(self.latent_dim, ))

        img = Input(shape=self.img_shape)

        d_in = concatenate([z, Flatten()(img)])



        model = Dense(1024)(d_in)

        model = LeakyReLU(alpha=0.2)(model)

        model = Dropout(0.5)(model)

        model = Dense(1024)(model)

        model = LeakyReLU(alpha=0.2)(model)

        model = Dropout(0.5)(model)

        model = Dense(1024)(model)

        model = LeakyReLU(alpha=0.2)(model)

        model = Dropout(0.5)(model)

        validity = Dense(1, activation="sigmoid")(model)



        return Model([z, img], validity)



    def train(self, epochs, batch_size=128, sample_interval=50):







        

        for epoch in range(epochs):

            for ii in tqdm(range(len(train_generator)), total=len(train_generator)):

                

                # Adversarial ground truths

                valid = np.ones((len(train_generator[ii][0]), 1))

                fake = np.zeros((len(train_generator[ii][0]), 1))

        

                z = np.random.normal(size=(len(train_generator[ii][0]), self.latent_dim))

                imgs_ = self.generator.predict(z)

                imgs = train_generator[ii][0]

                imgs = (imgs - 127.5) / 127.5

                

                z_ = self.encoder.predict(imgs)



                # Train the discriminator (img -> z is valid, z -> img is fake)

                d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)

                d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



                # ---------------------

                #  Train Generator

                # ---------------------



                # Train the generator (z -> img is valid and img -> z is is invalid)

                g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])



                # Plot the progress

                if (ii+1) % (len(train_generator)//2) == 0:

                    print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

        

                if (ii+1) % (len(train_generator)//2) == 0:

                    self.sample_interval(epoch)

                   

        for i in range(int(500)):

    

            z = np.random.normal(size=(20, self.latent_dim))

            gen_imgs = self.generator.predict(z)

            gen_imgs = 0.5 * gen_imgs + 0.5

    

            for j in range(20):

                img = image.array_to_img((gen_imgs[j, :,:,:]))

                img.save(os.path.join('../output_images/','generated_dog' + str(i) + '_'+ str(j) +'.png')) 

        plt.imshow(image.array_to_img(gen_imgs[3]))

        plt.show()

        plt.imshow(image.array_to_img(gen_imgs[15]))

        plt.show()

        plt.imshow(image.array_to_img(gen_imgs[5]))

        plt.show()



    def sample_interval(self, epoch):

        r, c = 5, 5

        z = np.random.normal(size=(25, self.latent_dim))

        gen_imgs = self.generator.predict(z)



        gen_imgs = 0.5 * gen_imgs + 0.5



        fig, axs = plt.subplots(r, c)

        cnt = 0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt, :,:,])

                axs[i,j].axis('off')

                cnt += 1

        #fig.savefig("../output_images/dog_%d.png" % epoch)

        plt.show()

        plt.close()

    

           



imgs = train_generator[30][0]

imgs_augment =train_generator_augment[30][0]

print(imgs.shape)

print(imgs_augment.shape)

imgs=np.concatenate((imgs,imgs_augment), axis=0)

print(imgs.shape)
if __name__ == '__main__':

    bigan = BIGAN()

    bigan.train(epochs=50, batch_size=batch_size, sample_interval=500)



    

import shutil

shutil.make_archive('images', 'zip', '../output_images')


