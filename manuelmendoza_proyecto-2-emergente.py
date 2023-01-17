import os

import numpy as np # linear algebra

import matplotlib.pyplot as plt # graphs

from tqdm import tqdm_notebook as tqdm # loading bar
from torchvision import datasets, transforms

import torch

from PIL import Image

import xml.etree.ElementTree as ET
img_dir = '../input/dogspictures/images/images/'

notes_dir = '../input/dogspictures/annotations/annotations/Annotation/'
rgb_counter = 0

rgba_counter = 0

rgba_images = []

for i, dog in tqdm(enumerate(os.listdir(img_dir)), total=len(os.listdir(img_dir))):

    img = Image.open(img_dir + dog)

    if(img.mode == 'RGB'):

        rgb_counter += 1

    elif(img.mode == 'RGBA'):

        rgba_images.append(dog)

        rgba_counter += 1

        

print("RGB images: " + str(rgb_counter))

print("RGBA images: " + str(rgba_counter))
# list for the final images

imgs_fixed = []

breeds = os.listdir(notes_dir)



for i, breed in tqdm(enumerate(breeds), total=len(breeds)):     # 1st loop for every breed of dog

    for dog in os.listdir(notes_dir + breed):                       # 2nd loop for every dog's picture inside the folder

        # Getting image and converting it if necessary

        img = Image.open(img_dir + dog + '.jpg')

        

        if(img.mode == 'RGBA'):

            img = img.convert('RGB')

            print("RGBA Image Converted!")

        

        # Getting annotations and coordinates

        tree = ET.parse(notes_dir + breed + '/' + dog)

        root = tree.getroot()

        objects = root.findall('object')

        for o in objects:

            bndbox = o.find('bndbox') 

            xmin = int(bndbox.find('xmin').text)

            ymin = int(bndbox.find('ymin').text)

            xmax = int(bndbox.find('xmax').text)

            ymax = int(bndbox.find('ymax').text)

        

        # Cropping image

        img_fixed = img.crop((xmin,ymin,xmax,ymax))

        imgs_fixed.append(img_fixed)
transform1 = transforms.Compose([transforms.Resize(64),

                                 transforms.CenterCrop(64)])
random_transforms = [transforms.RandomRotation(degrees=5)]

transform2 = transforms.Compose([transforms.Resize(64),

                                 transforms.CenterCrop(64),

                                 transforms.RandomHorizontalFlip(p=0.5),

                                 transforms.RandomApply(random_transforms, p=0.3)])
train_set1 = [] # 1st transform application

# train_set2 = [] # 2nd transform application



for i, img in tqdm(enumerate(imgs_fixed), total=len(imgs_fixed)):

        train_set1.append(transform1(imgs_fixed[i]))

        # train_set2.append(transform2(imgs_fixed[i]))



# Joining both datasets into a final one

X_train = train_set1



print(len(X_train))
for i in range(len(X_train)):

    X_train[i] = np.asarray(X_train[i])

    X_train[i] = ((X_train[i].astype(np.float32)) - 127.5)/127.5 # Normalization

    

X_train = np.asarray(X_train)
# Basic Keras

import keras

from keras.models import Model 

from keras.models import Sequential

# Layers Functions

from keras.layers import Input

from keras.layers import BatchNormalization

from keras.layers import Dense

from keras.layers import Reshape

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import UpSampling2D

from keras.layers import ReLU

from keras.layers.advanced_activations import LeakyReLU



from keras.initializers import RandomNormal

from keras.layers.core import Dropout
keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
def func_generator():

    generator = Sequential()



    # init weights for kernel, the starting shape and the input dimension

    init_kernel = RandomNormal(mean=0.0, stddev=0.02)

    init_shape = X_train.shape[1]*4*4

    init_dim = 128



    # 1st Layer

    generator.add(Dense(init_shape, kernel_initializer=init_kernel, input_dim=init_dim))

    generator.add(Reshape((4,4,X_train.shape[1]))) # Reshapes output

    # 1st Conv Layer

    generator.add(UpSampling2D())

    generator.add(Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer=init_kernel))

    generator.add(ReLU())

    # 2nd Conv Layer

    generator.add(UpSampling2D())

    generator.add(Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer=init_kernel))

    generator.add(ReLU())

    # 3rd Conv Layer

    generator.add(UpSampling2D())

    generator.add(Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer=init_kernel))

    generator.add(ReLU())

    # 4th Conv Layer

    generator.add(UpSampling2D())

    generator.add(Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer=init_kernel))

    generator.add(ReLU())



    # Output Layer

    generator.add(Conv2D(3, kernel_size=3, strides=1, padding='same', activation = 'tanh', kernel_initializer=init_kernel))

    generator.compile(loss = 'binary_crossentropy', optimizer = 'Adam')



    return generator
def func_discriminator():

    discriminator = Sequential()



    # Init values

    init_kernel = RandomNormal(mean=0.0, stddev=0.02)

    input_shape = (64, 64, 3)





    # 1st Layer

    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init_kernel, input_shape=input_shape))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.25))

    # 2nd Layer

    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init_kernel))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.25))

    # 3rd Layer

    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init_kernel))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.25))

    # 4th Layer

    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init_kernel))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.25))



    # Output Layer

    discriminator.add(Flatten())

    discriminator.add(Dense(1, activation = 'sigmoid', kernel_initializer = init_kernel))



    # Compile model

    discriminator.compile(loss = 'binary_crossentropy', optimizer = 'Adam')

    

    return discriminator
def func_gan_model(disc_model, gen_model, init_size):

    # Avoiding discriminator to learn

    disc_model.trainable = False

    

    # Generator output image

    noise = Input(shape = (init_size, ))

    fake_image = gen_model(noise)

    

    # Discriminator judgement

    disc_output = disc_model(fake_image)

    gan_model = Model(inputs=noise, outputs=disc_output)

    gan_model.compile(loss = 'binary_crossentropy', optimizer = 'Adam')

    

    return gan_model
def plot_generated_images(epoch, generator, examples = 25, dim = (5, 5)):

    generated_images = generator.predict(np.random.normal(0, 1, size = [examples, 128]))

    generated_images = ((generated_images + 1) * 127.5).astype('uint8')

        

    plt.figure(figsize = (12, 8))

    for i in range(generated_images.shape[0]):

        plt.subplot(dim[0], dim[1], i + 1)

        plt.imshow(generated_images[i], interpolation = 'nearest')

        plt.axis('off')

    plt.suptitle('Epoch %d' % epoch, x = 0.5, y = 1.0)

    plt.tight_layout()

    plt.savefig('dog_at_epoch_%d.png' % epoch)

    plt.show()

    

def graph_loss(d_f, d_r, g):

    plt.figure(figsize = (18, 12))

    plt.plot(d_f, label = 'Discriminator Fake Loss')

    plt.plot(d_r, label = 'Discriminator Real Loss')

    plt.plot(g, label = 'Generator Loss')

    plt.legend()

    plt.savefig('loss_plot.png')

    plt.show()
epochs=0

batch_size=128

# Creating models

gen_model = func_generator()

disc_model = func_discriminator()

gan_model = func_gan_model(disc_model, gen_model, 128)

    

# Batches

batch_count = X_train.shape[0] / batch_size

    

# List for losses

disc_fakeLosses = []

disc_realLosses = []

gen_losses = []

    

for e in range(epochs):

    print("Epoch N." + str(e))

        

    for i in tqdm(range(int(batch_count))):

        disc_batch_fakeLoss = []

        disc_batch_realLoss = []

            

        for j in range(2):

            # Creating Generator's input

            noise = np.random.randn(128 * batch_size)

            noise = noise.reshape((batch_size, 128))

                

            # Creating fake dataset

            X_fake = gen_model.predict(noise)

            Y_fake = np.zeros(batch_size)

            Y_fake[:] = 0

                

            # Training Discriminator with X_fake

            disc_model.trainable = True

            d_fakeLoss = disc_model.train_on_batch(X_fake, Y_fake)

                

            # Creating real dataset

            X_real = X_train[np.random.randint(0, X_train.shape[0], size = batch_size)]

            Y_real = np.zeros(batch_size)

            Y_real[:] = 0.9

                

            # Training Discriminator with X_real

            disc_model.trainable = True

            d_realLoss = disc_model.train_on_batch(X_real, Y_real)

                

            # Store Loss each iteration

            disc_batch_fakeLoss.append(d_fakeLoss)

            disc_batch_realLoss.append(d_realLoss)

            

        # Training Generator

        noise = np.random.randn(128 * batch_size)

        noise = noise.reshape((batch_size, 128))

            

        Y_gen = np.ones(batch_size)

        disc_model.trainable = False

        gen_loss = gan_model.train_on_batch(noise, Y_gen)

            

        # Storing Losses

        disc_fakeLosses.append(disc_batch_fakeLoss)

        disc_realLosses.append(disc_batch_realLoss)

        gen_losses.append(gen_loss)

        

    plot_generated_images(e, gen_model)
graph_loss(disc_fakeLosses, disc_realLosses, gen_losses)

plot_generated_images(100, gen_model)