import re

import os

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import layers

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import time

from IPython import display

import PIL



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)

    

print(tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path()

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [112, 112]
filenames = tf.io.gfile.glob(str(GCS_PATH + '/dataset/train/romanticism/*'))

filenames.extend(tf.io.gfile.glob(str(GCS_PATH + '/dataset/test/romanticism/*')))
IMG_COUNT = len(filenames)

print("Image count for training: " + str(IMG_COUNT))
train_ds = tf.data.Dataset.from_tensor_slices(filenames)



for f in train_ds.take(5):

    print(f.numpy())
# normalizing the images to [-1, 1]

def normalize(image):

  image = tf.cast(image, tf.float32)

  image = (image / 127.5) - 1

  return image
def decode_img(img):

  # convert the compressed string to a 3D uint8 tensor

  img = tf.image.decode_jpeg(img, channels=3)

  # Use `convert_image_dtype` to convert to floats in the [-1,1] range.

  img = normalize(img)

  # resize the image to the desired size.

  return tf.image.resize(img, IMAGE_SIZE)



def process_path(file_path):

    img = tf.io.read_file(file_path)

    img = decode_img(img)

    # convert the image to grayscale

    return tf.expand_dims(img[:, :, 0], axis=2)
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE).cache().batch(256)
image_batch = next(iter(train_ds))

image_batch.shape
def show_batch(image_batch):

    plt.figure(figsize=(10,10))

    for n in range(25):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n, :, :, 0], cmap='gray')

        plt.axis("off")
show_batch(image_batch)
EPOCHS = 25000

noise_dim = 100

num_examples_to_generate = 16



# We will reuse this seed overtime (so it's easier)

# to visualize progress in the animated GIF)

seed = tf.random.normal([num_examples_to_generate, noise_dim])
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*1024, use_bias=False, input_shape=(noise_dim,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((7, 7, 1024)))

    assert model.output_shape == (None, 7, 7, 1024) # Note: None is the batch size



    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, 7, 7, 512)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 14, 14, 256)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 28, 28, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 56, 56, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 112, 112, 1)



    return model
generator = make_generator_model()



noise = tf.random.normal([1, noise_dim])

generated_image = generator(noise, training=False)



plt.imshow(generated_image[0, :, :, 0], cmap='gray')
def make_discriminator_model():

    model = tf.keras.Sequential()

    

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[112, 112, 1]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model
discriminator = make_discriminator_model()

decision = discriminator(generated_image)

print (decision)
# helper function

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
# Notice the use of `tf.function`

# This annotation causes the function to be "compiled".

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



  fig = plt.figure(figsize=(4,4))



  for i in range(predictions.shape[0]):

      plt.subplot(4, 4, i+1)

      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

      plt.axis('off')



  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

  plt.show()
train(train_ds, EPOCHS)