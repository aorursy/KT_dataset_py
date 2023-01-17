# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xml.etree.ElementTree as ET # for parsing XML

import matplotlib

import matplotlib.pyplot as plt # to show images

from PIL import Image # to read images

import glob

from tqdm import tqdm

import random

import imageio

from scipy import stats

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import keras

import keras.backend as K

from keras import Sequential, Model, regularizers

from keras.layers import Conv2D, Conv2DTranspose, Dense, UpSampling2D, Input, Lambda, Concatenate

from keras.initializers import RandomNormal

from keras.optimizers import Adam

from keras.activations import relu, tanh

from keras.layers import PReLU, LeakyReLU, BatchNormalization, Reshape, Flatten, Dropout, Activation

from keras.preprocessing.image import ImageDataGenerator

from keras.losses import binary_crossentropy



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#defining file paths

image_path = '../input/all-dogs/all-dogs'

annot_path = '../input/annotation/Annotation/'
#getting all files

annot = glob.glob(f'{annot_path}/*/*')

images = glob.glob(f'{image_path}/*.jpg')

print(len(images),len(annot))
#loades bounding box data from XML files 

def get_bounding_box(annot):

    tree = ET.parse(annot)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox') # reading bound box

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        

    return (xmin,ymin,xmax,ymax)



bounding_box = {}

for i in tqdm(annot):

    img = i.split('/')[-1]

    bounding_box[img] = get_bounding_box(i)
#crops images using bounding box

def crop_image(img, box):

    im=Image.open(img)

    im=im.crop(box)

    return im
#looking at random cropped image

img_path = random.choice(images)

img = img_path.split('/')[-1].split('.')[0]

img = crop_image(img_path, bounding_box[img])

plt.imshow(img)
#process images

def process_image(img, box):

    im = crop_image(img, box)

    im = im.resize((64,64))

    im = np.array(im)

    im = (im.astype('float')-127.5)/127.5

    return im
#loading and processing all images

all_images = []

for img_path in images:

    img = img_path.split('/')[-1].split('.')[0]

    img = process_image(img_path, bounding_box[img])

    all_images.append(img)

all_images = np.array(all_images)
def generator_model():

    init = RandomNormal(mean=0.0, stddev=0.02)

    z = Input(shape=(4*4*64,))

    h = Reshape((4, 4, 64))(z)

    #out: (4,4,1024)

    h = Conv2DTranspose(1024, kernel_size=(4,4), strides=(1,1), padding='same', kernel_initializer=init, use_bias=False)(h)

    h = BatchNormalization(momentum=0.1)(h)

    h = PReLU()(h)

    h = Dropout(0.3)(h)

    #out: (8,8,512)

    h = Conv2DTranspose(512, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init, use_bias=False)(h)

    h = BatchNormalization(momentum=0.1)(h)

    h = PReLU()(h)

    h = Dropout(0.3)(h)

    #out: (16,16,256)

    h = Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init, use_bias=False)(h)

    h = BatchNormalization(momentum=0.1)(h)

    h = PReLU()(h)

    h = Dropout(0.3)(h)

    #out: (32,32,128)

    h = Conv2DTranspose(128, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init, use_bias=False)(h)

    h = BatchNormalization(momentum=0.1)(h)

    h = PReLU()(h)

    h = Dropout(0.3)(h)

    #out: (64,64,64)

    h = Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init, use_bias=False)(h)

    h = BatchNormalization(momentum=0.1)(h)

    h = PReLU()(h)

    h = Dropout(0.3)(h)

    #out: (64,64,3)

    h = Conv2DTranspose(3, kernel_size=(4,4), strides=(1,1), padding='same', kernel_initializer=init, use_bias=False)(h)

    h = BatchNormalization(momentum=0.1)(h)

    x = Activation('tanh')(h)

    model = Model(z,x, name='Generator')

    return model



def discriminator_model():

    init = RandomNormal(mean=0.0, stddev=0.02)

    x = Input(shape=(64,64,3))

    #out: (32,32,64)

    h = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)

    h = BatchNormalization()(h)

    h = LeakyReLU(0.2)(h)

    h = Dropout(0.4)(h)

    #out: (16,16,32)

    h = Conv2D(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(h)

    h = BatchNormalization()(h)

    h = LeakyReLU(0.2)(h)

    h = Dropout(0.4)(h)

    #out: (16,16,8)

    h = Conv2D(8, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(h)

    h = BatchNormalization()(h)

    h = LeakyReLU(0.2)(h)

    h = Dropout(0.4)(h)

    

    h = Flatten()(h)

    y = Dense(1, activation='sigmoid')(h)

    model = Model(x,y,name='Discriminator')

    return model



def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model

    model.add(generator)

    model.add(discriminator)

    return model
disc_opt = Adam()

discriminator = discriminator_model()

discriminator.compile(loss='binary_crossentropy',

                      optimizer=disc_opt,

                      metrics=['accuracy'])



# Build the Generator

generator = generator_model()

discriminator.trainable = False



# Build and compile GAN model with fixed Discriminator to train the Generator

gan = build_gan(generator, discriminator)

gan.compile(loss='binary_crossentropy', optimizer=Adam())
gan.summary()
# generate noisy labels

def noisy_labels(y):

    flip_idx = np.random.choice([i for i in range(y.shape[0])], size=int(y.shape[0]*0.05))

    y[flip_idx] = 1-y[flip_idx]

    return y



#prints a sample generated image

def sample_image(generator):

    z = np.random.normal(0, 1, (9, 1024))

    gen_imgs = generator.predict(z)

    gen_imgs = (gen_imgs + 1)/2

    return gen_imgs
batch_size = 16

y_real = np.zeros((batch_size, 1))

y_fake = np.ones((batch_size, 1))

aug = ImageDataGenerator()

aug = ImageDataGenerator(

    rotation_range=30,

    horizontal_flip=True,

    width_shift_range=0.2,

    height_shift_range=0.2,

    fill_mode="nearest")
#model training

losses = []

accuracies = []

iteration_checkpoints = []

iterations = 100000

sample_interval = 5000



for iteration in range(iterations):

        

        # random batch of real images

        idx = np.random.randint(0, all_images.shape[0], batch_size)

        imgs = all_images[idx]

        

        # data augmentation

        aug_images = []

        for X_batch, y_batch in aug.flow(imgs, y_real, batch_size=batch_size, shuffle=True):

            aug_images.append(X_batch)

            break

        aug_images = np.array(aug_images)

        aug_images = aug_images.reshape([batch_size,64,64,3])



        # label smoothing

        y_real_smooth = (y_real+0.2)-(np.random.random(y_real.shape)*0.2)

        y_fake_smooth = (y_fake-0.2)+(np.random.random(y_fake.shape)*0.3)

        

        # noisy labels

        y_real_noisy = noisy_labels(y_real_smooth)

        y_fake_noisy = noisy_labels(y_fake_smooth)

        

        # generate batch of fake images

        z = np.random.normal(0, 1, (batch_size, 1024))

        gen_imgs = generator.predict(z)



        # train Discriminator

        discriminator.trainable=True

        d_loss_real = discriminator.train_on_batch(aug_images, y_real_noisy)

        d_loss_fake = discriminator.train_on_batch(gen_imgs, y_fake_noisy)

        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)



        z = np.random.normal(0, 1, (batch_size, 1024))

        

        # train Generator

        discriminator.trainable=False

        g_loss = gan.train_on_batch(z, y_real)

        

        if (iteration + 1) % sample_interval == 0:



            # Save losses and accuracies so they can be plotted after training

            losses.append((d_loss, g_loss))

            accuracies.append(100.0 * accuracy)

            iteration_checkpoints.append(iteration + 1)



            # Output training progress

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %

                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))



            # Output a sample of generated image

            imgs = sample_image(generator)

            fig = plt.figure(figsize=(4,4))

            for i in range(1,10):

                fig.add_subplot(3,3,i)

                plt.imshow(imgs[i-1])

            plt.show()
if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

im_batch_size = 1

n_images=10000

for i_batch in range(0, n_images, im_batch_size):

    z = stats.truncnorm.rvs(-1,1,size=(im_batch_size, 1024))

    gen_imgs = generator.predict(z)

    gen_imgs = 0.5 * gen_imgs + 0.5

    for i_image in range(gen_imgs.shape[0]):

        imageio.imwrite(os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'), gen_imgs[i_image])

    idx = np.random.randint(0, all_images.shape[0], 4)

    imgs = all_images[idx]

    y_real = np.zeros((4,1))

    y_fake = np.ones((4,1))

    discriminator.trainable = True

    discriminator.train_on_batch(imgs, y_real)

    z = np.random.normal(0, 1, (4, 1024))

    gen_imgs = generator.predict(z)

    discriminator.train_on_batch(gen_imgs, y_fake)

    discriminator.trainable = False

    z = np.random.normal(0, 1, (4, 1024))

    gan.train_on_batch(z, y_real)



import shutil

shutil.make_archive('images', 'zip', '../output_images')