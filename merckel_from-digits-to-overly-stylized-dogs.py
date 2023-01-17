!pip install tensorflow-gpu==2.0.0-beta1
from __future__ import absolute_import, division, print_function, unicode_literals



import pandas as pd

import xml.etree.ElementTree as ET 

import zipfile 

import glob

import imageio

import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

from tensorflow.keras import layers

import time



from IPython import display

from PIL import Image 



import tensorflow as tf
ROOT = '../input/'

breeds = os.listdir(ROOT + 'annotation/Annotation/') 



idxIn = 0; 

namesIn = []

imagesIn = np.zeros((25000, 64, 64, 3))



for breed in breeds:

    for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):

        try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg')

        except: continue           

        tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)

        root = tree.getroot()

        objects = root.findall('object')

        for o in objects:

            bndbox = o.find('bndbox') 

            xmin = int(bndbox.find('xmin').text)

            ymin = int(bndbox.find('ymin').text)

            xmax = int(bndbox.find('xmax').text)

            ymax = int(bndbox.find('ymax').text)

            w = np.min((xmax - xmin, ymax - ymin))

            img2 = img.crop((xmin, ymin, xmin+w, ymin+w))

            img2 = img2.resize((64,64), Image.ANTIALIAS)

            imagesIn[idxIn,:,:,:] = np.asarray(img2)

            namesIn.append(breed)

            idxIn += 1
train_images = imagesIn.reshape(imagesIn.shape[0], 64, 64, 3).astype('float32')  

train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]



BUFFER_SIZE = 60000

BATCH_SIZE = 256



# Batch and shuffle the data

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((8, 8, 256)))

    assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size



    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, 8, 8, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 16, 16, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 32, 32, 32)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 64, 64, 3) 



    return model



generator = make_generator_model()
def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[64, 64, 3]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))    



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model



discriminator = make_discriminator_model()
# This method returns a helper function to compute cross entropy loss

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



EPOCHS = 300

noise_dim = 100

sqrt_num_examples_to_generate = 4



seed = tf.random.normal([sqrt_num_examples_to_generate * sqrt_num_examples_to_generate, noise_dim])
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



        # Save the model every 15 epochs

        if (epoch + 1) % 15 == 0:

            checkpoint.save(file_prefix = checkpoint_prefix)



        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



    # Generate after the final epoch

    display.clear_output(wait=True)

    generate_and_save_images(generator, seed)
def generate_and_save_images(model, test_input):

    # Notice `training` is set to False.

    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(15,15))

    for i in range(predictions.shape[0]):

        plt.subplot(sqrt_num_examples_to_generate, sqrt_num_examples_to_generate, i+1)

        plt.imshow((predictions[i, :, :, :] + 1.0) / 2.0)

        plt.axis('off')



    plt.savefig('image_latest_epoch.png')
%%time

train(train_dataset, EPOCHS)