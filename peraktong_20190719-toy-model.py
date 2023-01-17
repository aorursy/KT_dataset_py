# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import load_img

from keras.preprocessing.image import save_img

import sklearn

import os,matplotlib

import time

import matplotlib.pyplot as plt

from matplotlib.pylab import rc

import os

import h5py

import math

import os.path

import tensorflow as tf

import keras

import pickle

import pathlib

import random

from tensorflow.keras import layers

tf.enable_eager_execution()

tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Give credit to the tensorflow website: https://www.tensorflow.org/beta/tutorials/generative/dcgan

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_root = pathlib.Path("../input/all-dogs/")

all_image_paths = list(data_root.glob("*/*"))

print("Input pics number %d"%len(all_image_paths))

random.shuffle(all_image_paths)

image_count = len(all_image_paths)

# read Annotation (later)
# read dog classes:

# We don't need the name of the dog for now:

class_name_all = [str(i).split("_")[0].split("/")[-1] for i in all_image_paths]

class_name_unique = set(class_name_all)

print(len(class_name_unique),len(class_name_all))
## Define some pre-processing of data

# I want to use a resnet as an generator

size=224

# The inout shape is 224*224*3

# I will make a general version for this, for now I will fix that to 224*224 since it's the size of the Resnet in keras :)



# latent dimension, which is the input shape from generator:

latent_dim = 100

def preprocess_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_images(image, [size, size])

    image /= 255.0  # normalize to [0,1] range

    image = image*2-1  # normalize to [-1,1] range

    return image





def load_and_preprocess_image(path):

    image = tf.read_file(path)

    return preprocess_image(image)

# Image pre-processing:

# We pre-define how to read image and how to pre-process data

all_image_paths = [str(path) for path in all_image_paths]

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

# load data:

image_ds = path_ds.map(load_and_preprocess_image)

# use cache to boost up the speed [This is experimental feature and can use autotune in tf2.0]

# disable cache if you do not have enough RAM

#ds = image_ds.cache()

ds = image_ds

"""

ds = ds.apply(

  tf.data.experimental.shuffle_and_repeat(buffer_size=64))



"""



# Define generator: upsamepl until you reach size of the fig: 56-112-224

def Generator():

    model = tf.keras.Sequential()

    # The input shape should be the latent_dim,

    

    # Add 28*28*256 neurons for the first layer

    model.add(layers.Dense(int(size/4)*int(size/4)*256, use_bias=False, input_shape=(latent_dim,)))

    # Add batch normalization to avoid over fitting. You can also use dropout here:

    model.add(layers.BatchNormalization())

    # By default the leaky relu alpha=0.3, you can adjust it.

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((int(size/4),int(size/4), 256)))

    # assert for debugging :)

    

    assert model.output_shape == (None, int(size/4),int(size/4), 256) # Note: None is the batch size

    

    # Add Transpose layer since we need to go from output to input

    # Use (1,1) stride

    # Here we use (5,5) filter size and 128 filters, use same padding to make sure the output is the same as input

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, int(size/4), int(size/4), 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    # continue to lower the resolution for the neural net work.This time set stride=2 



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, int(size/2), int(size/2), 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    # same here, but only one filter and the return shape is the same as input

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, size, size, 3)



    return model
# summary of the generator

generator = Generator()

generator.summary()
# check our generator:









# batch size=3

noise = tf.random.normal([3,latent_dim])

generated_image = generator(noise, training=False)



plt.imshow(generated_image[0,:, :, 0])
# Define discriminator:

# A simple discriminator 

# Remember the output of the discriminator is a classifier: (True/Fake)

def discriminator():

    model = tf.keras.Sequential()

    # first layer should be a Dense layer: Shape is the same as the shape from generator

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[size, size, 3]))

    # Use leaky relu as activation function to avoid "always positive" from relu 

    # default alpha=0.3

    model.add(layers.LeakyReLU())

    # default =0.5, here we use 0.3

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model

# check our discriminator from the generated fake image

discriminator = discriminator()

decision = discriminator(generated_image)

# The values are different since it's from different random seed, and it means our model is correct :)

print (decision)
discriminator.summary()
## Let's build our GAN !!



# loss for discriminator: it's binary cross-entropy since we only need to tell yes or no:

# set from_logits=True to avoid probability conversion :)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def discriminator_loss(real_output, fake_output):

    # compare real_image_output 

    # Here 1 is real, so we compare "real" for real output" to evaluate how well the discriminator can tell it's real

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    # compare fake_image_output: Zero means false and vice versa

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    # tot

    total_loss = real_loss + fake_loss

    return total_loss



# generator loss:

# Let's tell how well the generator can "trick" the discriminator

def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)





# define optimizer for both the generator and discriminator: use adam

generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
# check points:

checkpoint_dir = 'checkpoints_1.ckpt'

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)

# epochs and batch_size

n_epochs = 25

batch_size=32



noise_dim = latent_dim

ds = ds.batch(batch_size=batch_size).prefetch(buffer_size=64)

print("Doing image batch")

#image_batch = next(iter(ds))

N_step=int(image_count/batch_size)
ds
def train(image_batch):

    # print("Doing %d epoch of %d epoch" % (epoch, n_epochs))



    # GradientTape: automatically calculate the gradient of a computation with respect to its input variables

    # The generator start with noise

    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)



        real_output = discriminator(image_batch, training=True)

        fake_output = discriminator(generated_images, training=True)



        gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(real_output, fake_output)



    # The gradient

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



    # optimize the gradient

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    

    return gen_loss,disc_loss

        









    

print("Start training")

## Let's train it:



for epoch in range(n_epochs):

    print("Doing %d of %d epoch"%(epoch,n_epochs))

    start = time.time()

    count=0

    for i in iter(ds):

        

        gen_loss,disc_loss=train(i)

    

        if count%50==0:

            print("%d of %d step"%(count,N_step))

            print("Generator loss=%.2f Discriminator loss=%.2f"%(gen_loss,disc_loss))

        

        count+=1



    #save:





print("Finish training!")

#checkpoint_prefix = "toy.ckpt"

#checkpoint.save(file_prefix = checkpoint_prefix)
print("Done")
# predict

n_test=10000

batch_test = 100

count=0

os.system("mkdir %s"%("images_out"))

for i in range(n_test//batch_test):

    print("Doing %d of %d"%(batch_test*i,n_test))

    noise = tf.random.normal([batch_test,latent_dim])

    generated_images_i = generator(noise, training=False)

    image_temp = tf.image.resize_images(generated_images_i, [64, 64])

    # save image:

    for j in range(batch_test):

        save_img("images_out/"+'{}.JPEG'.format(count), tf.image.resize_images(generated_images_i[j,:,:,:], [64,64]))

        count+=1

        

    

    

    

# restore



# restored_model = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
"""

batch_test = 100

os.system("mkdir -p %s"%("images"))

noise = tf.random.normal([batch_test,latent_dim])

generated_images_i = generator(noise, training=False)

generated_images_i = tf.image.resize_images(generated_images_i, [64,64])

save_img("images"+'t1.JPEG', generated_images_i[0,:,:,:])



image = tf.read_file('t1.JPEG')

image = tf.image.decode_jpeg(image, channels=3)



"""
import shutil

shutil.make_archive('images', 'zip', 'images_out')

# os.system("rm -r images")
os.system("rm -r images_out")

print("All finished")
ls