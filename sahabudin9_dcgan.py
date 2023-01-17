# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf



# To generate GIFs

!pip install -q imageio



import glob

import imageio

import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

from tensorflow.keras import layers

import time



from IPython import display

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Deep Convolution Adversarial Networks 

#Genrator becomes better at creating the real images

#while discriminator becomes the telling them apart,

# this process becomes equillibrium when discriminator, 

#can no longer distinguish between real and fake images

# In this projct we see that how computer can learn to create images.
# we use MNIST Dataset of handwritten digits.





(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#Prepare the dataset

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000

BATCH_SIZE = 256



# Batch and shuffle the data

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#Define the Generator that craetes an image 28x28

def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((7, 7, 256)))

    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size



    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 14, 14, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 28, 28, 1)



    return model
#Use the (so far untrained) generator to create an image

generator = make_generator_model()



noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)



plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#Define the Discriminator as CNN-based image classifier

def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[28, 28, 1]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model
#Use the (so far untrained) discriminator to classify the generated images as real or fake

#The model will be trained to output positive values for real images, and negative values for fake images.

discriminator = make_discriminator_model()

decision = discriminator(generated_image)

print (decision)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
#Define the optimizers separately.

generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#Save checkpoints

checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
#Define the training loop

EPOCHS = 50

noise_dim = 100

num_examples_to_generate = 16



# We will reuse this seed overtime to visualize progress in the animated GIF

seed = tf.random.normal([num_examples_to_generate, noise_dim])
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
def train(dataset, epochs):

  for epoch in range(epochs):

    start = time.time()



    for image_batch in dataset:

      train_step(image_batch)



    # Produce images for the GIF as we go

    display.clear_output(wait=True)

    generate_and_save_images(generator,

                             epoch + 1,

                             seed)



    # Save the model every 15 epochs

    if (epoch + 1) % 15 == 0:

      checkpoint.save(file_prefix = checkpoint_prefix)



    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



  # Generate after the final epoch

  display.clear_output(wait=True)

  generate_and_save_images(generator,

                           epochs,

                           seed)
#Generate and save images

def generate_and_save_images(model, epoch, test_input):

  # Notice `training` is set to False.

  # This is so all layers run in inference mode (batchnorm).

  predictions = model(test_input, training=False)



  fig = plt.figure(figsize=(4,4))



  for i in range(predictions.shape[0]):

      plt.subplot(4, 4, i+1)

      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

      plt.axis('off')



  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

  plt.show()
#Train the model



train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
def display_image(epoch_no):

  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))



display_image(EPOCHS)