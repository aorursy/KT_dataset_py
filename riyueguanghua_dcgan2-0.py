!pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import glob

import os
os.listdir('../input/anime-faces/data/data')[:10]
print('Tensorflow version: {}'.format(tf.__version__))
image_path = glob.glob('../input/anime-faces/data/data/*.png')
len(image_path)
def load_preprosess_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(image, channels=3)

#    image = tf.image.resize_with_crop_or_pad(image, 256, 256)

    image = tf.cast(image, tf.float32)

    image = (image / 127.5) - 1

    return image
image_ds = tf.data.Dataset.from_tensor_slices(image_path)
AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
image_ds
BATCH_SIZE = 64

image_count = len(image_path)
image_ds = image_ds.shuffle(image_count).batch(BATCH_SIZE)

image_ds = image_ds.prefetch(AUTOTUNE)
def generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((8, 8, 256)))   #8*8*256



    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())             #8*8*128



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())             #16*16*128

    

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())             #32*32*32

    



    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

                                              #64*64*3

    

    return model
generator = generator_model()



noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)



plt.imshow((generated_image[0, :, :, :3] + 1)/2)
def discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[64, 64, 3]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))      # 32*32*32



    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))      # 16*16*64

    

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

#    model.add(layers.Dropout(0.3))      # 8*8*128

    

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())       # 4*4*256



    model.add(layers.GlobalAveragePooling2D())

    

    model.add(layers.Dense(1024))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    model.add(layers.Dense(1))



    return model
discriminator = discriminator_model()

decision = discriminator(generated_image)

print(decision)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-5)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
EPOCHS = 800

noise_dim = 100

num_examples_to_generate = 4



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
def generate_and_save_images(model, epoch, test_input):

  # Notice `training` is set to False.

  # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)



    fig = plt.figure(figsize=(6, 6))



    for i in range(predictions.shape[0]):

        plt.subplot(2, 2, i+1)

        plt.imshow((predictions[i, :, :, :] + 1)/2)

        plt.axis('off')



#    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()
def train(dataset, epochs):

    for epoch in range(epochs):

        for image_batch in dataset:

            train_step(image_batch)

#            print('.', end='')

#        print()

        if epoch%10 == 0:

            generate_and_save_images(generator,

                                     epoch + 1,

                                     seed)





    generate_and_save_images(generator,

                           epochs,

                           seed)
train(image_ds, EPOCHS)