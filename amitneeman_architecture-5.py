from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.__version__
# To generate GIFs

#!pip install imageio
import glob

#import imageio

import matplotlib.pyplot as plt

import numpy as np

import os

import cv2

import PIL

from tensorflow.keras import layers

import time

import pathlib

import IPython.display as display

from PIL import Image

from tqdm import tqdm

import random



from IPython import display



tf.compat.v1.enable_eager_execution()
TARGET_IMG_WIDTH = 320

TARGET_IMG_HEIGHT = 140
BUFFER_SIZE = 4415

BATCH_SIZE = 32
datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_dataset = datagen.flow_from_directory('/kaggle/input/fishes/', target_size=(140,320), batch_size=BATCH_SIZE, class_mode=None)

#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_dataset)
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(16*7*512, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU(0.2))



    model.add(layers.Reshape((7, 16, 512)))

    assert model.output_shape == (None, 7, 16, 512)  

                                                    



    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(5, 5), padding='same', use_bias=False))

    assert model.output_shape == (None, 35,80, 256)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU(0.2))



    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 70, 160, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU(0.2))



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, 70, 160, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU(0.2))



    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 140, 320, 3)



    return model
generator = make_generator_model()



noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)



print("First generation:")

plt.imshow(generated_image[0, :, :, 0])
generator.summary()
def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[140, 320, 3]))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model
discriminator = make_discriminator_model()

discriminator.summary()
# This method returns a helper function to compute cross entropy loss

cross_entropy_gen = tf.keras.losses.BinaryCrossentropy(from_logits=True)

cross_entropy_disc = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy_disc(tf.zeros_like(real_output), real_output)

    fake_loss = cross_entropy_disc(tf.ones_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    gen_loss = cross_entropy_gen(tf.zeros_like(fake_output), fake_output)

    return gen_loss
generator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
#checkpoint_dir = '/content/gdrive/My Drive/manager_checkpoints_arch4'

#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

 #                                discriminator_optimizer=discriminator_optimizer,

  #                               generator=generator,

   #                              discriminator=discriminator)

#manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
start_epoch = 0

EPOCHS = 600

noise_dim = 100

num_examples_to_generate = 8



# We will reuse this seed overtime (so it's easier)

# to visualize progress in the animated GIF)

seed = tf.random.normal([num_examples_to_generate, noise_dim])
# Notice the use of `tf.function`

# This annotation causes the function to be "compiled" - fast running

@tf.function

def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)

      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)

      disc_loss = discriminator_loss(real_output, fake_output)



    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss,disc_loss
def train(dataset, epochs):

  for epoch in range(start_epoch,epochs):

    print("strating epoch {}".format(epoch))

    start = time.time()

    for i in range(len(dataset)):

        image_batch = dataset.next()

        train_step((image_batch / 127.5 )-1)



    if (epoch + 1) % 1 == 0:

      #print("creating checkpoint... {}".format(epoch + 1))

        discriminator.save_weights('discriminator.h5')

        generator.save_weights('generator.h5')

      #manager.save()

        display.clear_output(wait=True)

        #generate_and_save_images(generator,epoch + 1,seed)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



  # Generate after the final epoch

  display.clear_output(wait=True)

  generate_and_save_images(generator,

                           epochs,

                           seed)
def generate_and_save_images(model, epoch, test_input):

  # Notice `training` is set to False.

  # This is so all layers run in inference mode (batchnorm).

  predictions = model(test_input, training=False)



  fig = plt.figure(figsize=(30,15))



  for i in range(predictions.shape[0]):

      plt.subplot(4, 4, i+1)

      plt.imshow(((np.uint8(predictions * 127.5 + 127.5))[i]))

      #predictions[i, :, :, 0] * 127.5 + 127.5

      plt.axis('off')



  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

  plt.show()
def train_and_checkpoint():

  try:

    #discriminator.load_weights('/content/gdrive/My Drive/model/discriminator.h5')

    #generator.load_weights('/content/gdrive/My Drive/model/generator.h5')

    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    #status = checkpoint.restore(manager.latest_checkpoint).assert_consumed()

    print(status)

  except:

    print("Starting from scratch")

  train(train_dataset, EPOCHS)

train_and_checkpoint()
noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)

plt.imshow(((np.uint8(generated_image * 127.5 + 127.5))[0]))

print(discriminator(generated_image, training=False))

#print(discriminator(train_images[0], training=False))

#bla = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(1)

#for image_batch in bla:

 # print(discriminator(image_batch))
#discriminator.load_weights('/content/gdrive/My Drive/model/disc_original.h5')

#generator.load_weights('/content/gdrive/My Drive/model/gen_original.h5')
#checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()