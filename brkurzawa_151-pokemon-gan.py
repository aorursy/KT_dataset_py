# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os # Handle working with files and directories

import tensorflow as tf # Loading images, storing them in tensors

import cv2

import PIL

from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 933120000

import matplotlib.pyplot as plt
pokemon_dir = '../input/pokemon/pokemon'
# Convert an image to a jpeg

def convert_to_jpg(img_path):

    # Convert png to jpeg

    img = Image.open(img_path)

    if img.mode == 'RGBA':

        img.load()

        background = Image.new("RGB", img.size, (0,0,0))

        background.paste(img, mask=img.split()[3])

        img = np.array(background)

    else:

        img = img.convert('RGB')

        img = np.array(img)

    

    return img

        

# Resize image to 128x128

def resize_img(img):

    img = cv2.resize(img, (128,128))

    return img



# Normalize pixel values from -1 to 1

def normalize_img(img):

    img = img / 127.5 - 1

    return img



# Open an image, convert to jpeg, resize if needed

def open_convert(img_path):

    # png

    if img_path[-4:] == '.png':

        img = convert_to_jpg(img_path)

    # jpeg

    else:

        img = Image.open(img_path)

        img = img.convert('RGB')

        img = np.array(img)



        

    # Convert to 128x128

    img = resize_img(img)

    

    # Normalize img

    img = normalize_img(img)

    

    # Return resized img

    return img



# Test

img = open_convert('../input/pokemon/pokemon/Aerodactyl/00000048.png~original')

# img = Image.fromarray(img, 'RGB')

# img.save('my.png')

plt.imshow(img)

plt.show()
# Contain images and labels

images = []

labels = []



# How many images per pokemon to load

images_per_pokemon = 5



# Keep track of current iteration

count = 0

# Iterate through each pokemon folder

for pkmn in os.listdir(pokemon_dir):

    pkmn_dir = os.path.join(pokemon_dir, pkmn)

    

    # Current number of images loaded for this pokemon

    curr_imgs = 0

    

    # Add each image to the list

    for img in sorted(os.listdir(pkmn_dir)):

        # Attempt to add image and label to list

        try:

            images.append(open_convert(os.path.join(pkmn_dir, img)))

            labels.append(pkmn)

        except (ValueError, OSError):

            continue

        count += 1

        if count % 1000 == 0:

            print('Current iteration: ' + str(count))

            

        # Increment num images loaded

        curr_imgs += 1

        if curr_imgs >= images_per_pokemon:

            break
plt.imshow(images[123])

plt.show()
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D, Reshape, Input

from keras.layers.advanced_activations import LeakyReLU

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau



input_shape = (images[0].shape)



latent_dim = 100



def make_gen():

    Gen = Sequential()



    Gen.add(Dense(256, input_dim=latent_dim))

    Gen.add(LeakyReLU(alpha=0.2))

    Gen.add(BatchNormalization(momentum=0.8))

    Gen.add(Dense(512))

    Gen.add(LeakyReLU(alpha=0.2))

    Gen.add(BatchNormalization(momentum=0.8))

    Gen.add(Dense(1024))

    Gen.add(LeakyReLU(alpha=0.2))

    Gen.add(BatchNormalization(momentum=0.8))

    Gen.add(Dense(np.prod(input_shape), activation='tanh'))

    Gen.add(Reshape(input_shape))



    Gen.summary()

    

    noise = Input(shape=(latent_dim,))

#     noise = Input(shape=input_shape)

    img = Gen(noise)

    

    return Model(noise, img)

    

def make_discr():

    Discr = Sequential()



    Discr.add(Conv2D(128, (3, 3), strides=(2, 2), input_shape=input_shape))

    Discr.add(LeakyReLU(alpha=0.2))

    Discr.add(AveragePooling2D(pool_size = (4, 4)))

    Discr.add(Flatten())

    Discr.add(Dense(units=512, activation='relu'))

    Discr.add(Dense(units=1, activation='sigmoid'))

    

    Discr.summary()

    

    img = Input(shape=input_shape)

    validity = Discr(img)

    

    return Model(img, validity)
optimizer = Adam(0.0002, 0.5)



Discr = make_discr()

Discr.compile(loss='binary_crossentropy',

             optimizer=optimizer,

             metrics=['accuracy'])



Gen = make_gen()



z = Input(shape=(latent_dim,))

# z = Input(shape=input_shape)

img = Gen(z)



Discr.trainable = False



validity = Discr(img)



Combined = Model(z, validity)

Combined.compile(loss='binary_crossentropy', optimizer=optimizer)



batch_size = 128



epochs = 45000



sample_interval = 1000



valid = np.ones((batch_size, 1))

fake = np.zeros((batch_size, 1))



# images = np.array(images)

# images = np.expand_dims(images, axis=3)



def sample_images(epoch):

    r, c = 5, 5

    noise = np.random.normal(0, 1, (r * c, latent_dim))

#     noise = np.random.randn(r*c, 128, 128, 3)



    gen_imgs = Gen.predict(noise)

    

    gen_imgs = 0.5 * gen_imgs + 0.5

    

#     for cnt in range(len(gen_imgs)):

#         imgname = 'epoch%dimg%d' % (epoch, cnt)

#         gen_imgs.to_csv(imgname, index=False)

    

    fig, axs = plt.subplots(r, c)

    cnt = 0

    for i in range(r):

        for j in range(c):

            axs[i,j].imshow(gen_imgs[cnt, :,:,:])

            axs[i,j].axis('off')

            cnt += 1

#     fig.savefig("/%d.png" % epoch)

#     plt.close()

    plt.show()

    

def sample_big_images(epoch):

    noise = np.random.normal(0, 1, (5, latent_dim))

#     noise = np.random.randn(5, 128, 128, 3)

    

    gen_imgs = Gen.predict(noise)

    

    gen_imgs = 0.5 * gen_imgs + 0.5

    

    # Show 5 images

    for i in range(5):

        plt.imshow(gen_imgs[i, :, :, :])

        plt.show()





for epoch in range(epochs):

    idx = np.random.randint(0, len(images), batch_size)

    imgs = np.array([images[j] for j in idx])

    

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

#     noise = np.random.randn(batch_size, 128, 128, 3)

    

    gen_imgs = Gen.predict(noise)

    

    d_loss_real = Discr.train_on_batch(imgs, valid)

    d_loss_fake = Discr.train_on_batch(gen_imgs, fake)

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

#     noise = np.random.randn(batch_size, 128, 128, 3)



    g_loss = Combined.train_on_batch(noise, valid)

    

    if epoch % sample_interval == 0:

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        sample_images(epoch)

        sample_big_images(epoch)

    
for i in range(5):

    sample_images(epoch)

    sample_big_images(epoch)