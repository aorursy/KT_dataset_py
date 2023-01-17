
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
noise_dim = 256
n = 48
features_dim = 256
num_examples_to_generate = 16

def ginitializer():
    return None  # tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.008, seed=None)


def dinitializer():
    return None  # tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.008, seed=None)


def make_generator_model():
    model = tf.keras.Sequential(name="generator")
    model.add(layers.Dense(35 * 40 * n, use_bias=False, input_shape=(noise_dim,), kernel_initializer=ginitializer()))
    model.add(layers.Reshape((35, 40, n)))

    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.UpSampling2D((1, 2)))
    model.add(layers.ELU())

    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.UpSampling2D())
    model.add(layers.ELU())

    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.UpSampling2D())
    model.add(layers.ELU())

    model.add(layers.Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='tanh',
                            kernel_initializer=dinitializer()))
    return model


def make_discriminator_encoder():
    model = tf.keras.Sequential(name="encoder")

    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer(),
                            input_shape=(140, 320, 3)))
    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.MaxPooling2D())
    model.add(layers.ELU())

    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.MaxPooling2D())
    model.add(layers.ELU())

    model.add(layers.Conv2D(3 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(3 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.MaxPooling2D())
    model.add(layers.ELU())

    model.add(layers.Flatten())
    model.add(layers.Dense(features_dim))

    return model


def make_discriminator_decoder():
    model = tf.keras.Sequential(name="decoder")
    model.add(layers.Dense(35 * 40 * n, use_bias=False, input_shape=(features_dim,), kernel_initializer=dinitializer()))
    model.add(layers.Reshape((35, 40, n)))

    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.UpSampling2D((1, 2)))
    model.add(layers.ELU())

    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.UpSampling2D())
    model.add(layers.ELU())

    model.add(layers.Conv2D(n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.Conv2D(2 * n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=dinitializer()))
    model.add(layers.UpSampling2D())
    model.add(layers.ELU())

    model.add(layers.Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='tanh',
                            kernel_initializer=dinitializer()))
    return model


def make_discriminator():
    input = layers.Input((140, 320, 3))
    x = make_discriminator_encoder()(input)
    x = make_discriminator_decoder()(x)
    return tf.keras.Model(input, x)
@tf.function
def train_real(real_input):
    with tf.GradientTape() as discriminator_tape:
        real_output = discriminator(real_input, training=True)
        disc_loss = loss_function(real_input, real_output)

    gradients = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    return disc_loss


@tf.function
def train_fake(kt):
    noise = tf.random.uniform([BATCH_SIZE, noise_dim], minval=-1)
    generator.trainable = False
    fake_input = generator(noise, training=True)
    with tf.GradientTape() as discriminator_tape:
        fake_output = discriminator(fake_input, training=True)
        disc_loss = -float(kt) * loss_function(fake_input, fake_output)

    gradients = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    generator.trainable = True

    return disc_loss


@tf.function
def train_gen():
    noise = tf.random.uniform([BATCH_SIZE, noise_dim], minval=-1)

    with tf.GradientTape() as gen_tape:
        generated = generator(noise, training=True)
        discriminator.trainable = False
        output = discriminator(generated, training=True)
        discriminator.trainable = True
        gen_loss = loss_function(generated, output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss


def train(epochs, start_epoch=0):
    global k
    for epoch in range(start_epoch, epochs):
        start = time.time()
        print("Starting epoch number %d" % (epoch + 1))
        disc_loss, gen_loss, real_loss, fake_loss = 0, 0, 0, 0
        for i in range(len(data_it)):
            image_batch = data_it.next()
            cur_real_loss = train_real(image_batch)
            cur_fake_loss = train_fake(tf.Variable(k))
            cur_disc_loss = cur_real_loss + cur_fake_loss
            cur_gen_loss = train_gen()

            # Update k
            k = float(k + kLambda * (gamma * float(cur_real_loss) - float(cur_gen_loss)))
            k = max(min(k, 1), epsilon)

            disc_loss += cur_disc_loss
            gen_loss += cur_gen_loss
            real_loss += cur_real_loss
            fake_loss += cur_fake_loss

        real_loss /= len(data_it)
        fake_loss /= len(data_it)
        disc_loss /= len(data_it)
        gen_loss /= len(data_it)
        Mglobal = float(real_loss) + abs(float(gamma * real_loss - gen_loss))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print(f"disc_loss: {float(disc_loss):.4f} real={float(real_loss):.4f},"
              f" fake={float(fake_loss):.4f}, kt={k:.4f}")
        print(f"gen_loss: {float(gen_loss):.4f}")
        print(f"global mode: {Mglobal}")
        if (epoch + 1) % 20 == 0:
            print("Checkpoint...")
            checkpoint.save(file_prefix=checkpoint_prefix)
        generate_and_save_images(generator, epoch + 1, seed, image_batch)
        print('Saved Images :)')

        print('---------------------------')

    # Generate after the final epoch
    # display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed,
                             image_batch)


def generate_and_save_images(model, epoch, test_input, image_batch):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    after = discriminator(predictions)

    fig = plt.figure(figsize=(32, 16))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) * 0.5)
        plt.axis('off')
    try:
        plt.savefig('images/{:04d}_generated.png'.format(epoch))
    except:
        pass
    plt.close(fig)

    fig = plt.figure(figsize=(32, 16))
    for i in range(after.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((after[i] + 1) * 0.5)
        plt.axis('off')
    try:
        plt.savefig('images/{:04d}_after.png'.format(epoch))
    except:
        pass
    plt.close(fig)

    y_hat = (discriminator(image_batch) + 1) * 0.5
    for i in range(image_batch.shape[0]):
        original = ((image_batch[i, :, :, :] + 1) * 0.5)
        after = y_hat[i, :, :, :]
        plt.imsave(f'output_images/image{i}-original.png', original.clip(0, 1))
        plt.imsave(f'output_images/image{i}-after.png', after.numpy().clip(0, 1))
    plt.close()


BATCH_SIZE = 16
EPOCHS = 1000
noise_dim = 256
n = 48
features_dim = 256
num_examples_to_generate = 16
lr = 1e-4

datagen = ImageDataGenerator(preprocessing_function=lambda image: (image - 127.5) / 127.5)
data_it = datagen.flow_from_directory('../fishDataSets/', target_size=(140, 320), class_mode=None,
                                      batch_size=BATCH_SIZE)

epsilon = 1e-5
gamma = .7
kLambda = 3e-3
k = epsilon
generator = make_generator_model()
generator_optimizer = tf.keras.optimizers.Adam()

discriminator = make_discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = '../input/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
noise = tf.random.uniform([num_examples_to_generate, noise_dim], minval=-1, maxval=1)
output = generator(noise, training=False)
after = discriminator(output)

fig = plt.figure(figsize=(8, 4))
for i in range(output.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((output[i] + 1) * 0.5)
    plt.axis('off')
print("Generated:")
plt.show(fig)
plt.close(fig)

print("after autoencoder")
fig = plt.figure(figsize=(8, 4))
for i in range(after.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((after[i] + 1) * 0.5)
    plt.axis('off')
plt.show(fig)
plt.close(fig)