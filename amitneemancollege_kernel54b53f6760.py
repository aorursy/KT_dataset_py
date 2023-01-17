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

import keras.backend as K



from IPython import display



tf.compat.v1.enable_eager_execution()
TARGET_IMG_WIDTH = 320

TARGET_IMG_HEIGHT = 140
BUFFER_SIZE = 4415

BATCH_SIZE = 32
datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_dataset = datagen.flow_from_directory('/kaggle/input/fish-2/fishDataSets', target_size=(140,320), batch_size=BATCH_SIZE, class_mode=None)

#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_dataset)
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(80 * 35 * 256, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())  # Normalize and scale inputs or activations. See remark bellow

    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((35, 80, 256)))



    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))



    return model
generator = make_generator_model()



noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)



print("First generation:")

plt.imshow(generated_image[0, :, :, 0])
generator.summary()
def make_discriminator_model():

    model = tf.keras.Sequential()



    model.add(layers.Conv2D(6, (3, 3), strides=(1, 1), padding='same',input_shape=[140, 320, 3]))

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Flatten())



    model.add(layers.Dense(1))



    return model
discriminator = make_discriminator_model()

discriminator.summary()
loss =  tf.keras.losses.MeanSquaredError()

#loss =  tf.keras.losses.Hinge()
def discriminator_loss(real_output,mean_real,fake_output,mean_fake):

    disc_loss = loss((real_output - mean_fake),tf.ones_like(real_output)) + loss((fake_output - mean_real + tf.ones_like(fake_output)),tf.zeros_like(fake_output))

    #disc_loss = loss(tf.ones_like(real_output),real_output - mean_fake) + loss(tf.ones_like(fake_output) * -1,fake_output - mean_real)

    return (disc_loss / 2)
def generator_loss(real_output,mean_real,fake_output,mean_fake):

    gen_loss = loss((fake_output - mean_real),tf.ones_like(fake_output)) + loss((real_output - mean_fake +  tf.ones_like(real_output)),tf.zeros_like(real_output)) 

    #gen_loss = loss(tf.ones_like(fake_output),fake_output - mean_real) + loss(tf.ones_like(real_output) * - 1,real_output  - mean_fake)

    return (gen_loss /2 )
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-07)

discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
#checkpoint_dir = '/content/gdrive/My Drive/manager_checkpoints_arch4'

#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

 #                                discriminator_optimizer=discriminator_optimizer,

  #                               generator=generator,

   #                              discriminator=discriminator)

#manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
start_epoch = 0

EPOCHS = 450

noise_dim = 100

num_examples_to_generate = 8

disc_losses = []

gen_losses = []



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

      

      # Calculate the means

      mean_fake = tf.keras.backend.mean(fake_output,keepdims=True)

      mean_real = tf.keras.backend.mean(real_output,keepdims=True)

    

      gen_loss = generator_loss(real_output,mean_real,fake_output,mean_fake)

      disc_loss = discriminator_loss(real_output,mean_real,fake_output,mean_fake)



    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss,disc_loss
def train(dataset, epochs):

  for epoch in range(start_epoch,epochs):

    print("strating epoch {}".format(epoch))

    start = time.time()

    current_disc_loss = 0

    current_gen_loss = 0

    

    for i in range(len(dataset)):

        image_batch = dataset.next()

        gen_loss,disc_loss = train_step((image_batch / 127.5 )-1)

        current_disc_loss += disc_loss

        current_gen_loss += gen_loss

        

    if (epoch + 1) % 1 == 0:

      #print("creating checkpoint... {}".format(epoch + 1))

        #discriminator.save_weights('discriminator.h5')

        #generator.save_weights('generator.h5')

      #manager.save()

        display.clear_output(wait=True)

        generate_and_save_images(generator,epoch + 1,seed)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        disc_losses.append(current_disc_loss / len(dataset))

        gen_losses.append(current_gen_loss / len(dataset))



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

discriminator.save_weights('discriminator.h5')

generator.save_weights('generator.h5')
display.clear_output(wait=True)

X = list(range(1, EPOCHS + 1))

plt.plot(X, disc_losses, color='red', label='disc')

plt.plot(X, gen_losses, color='blue', label='gen')

plt.legend()