import tensorflow as tf

from tensorflow.keras import layers

import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras import layers

import time

from PIL import Image



from keras.datasets.fashion_mnist import load_data



print(tf.__version__)



TPU_used = False



if TPU_used:

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection

        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    except ValueError:

        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')



    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
np.random.seed(1337)

num_classes = 10



epochs = 30

latent_dim = 128



adam_lr = 0.0002

adam_beta_1 = 0.5
batch_size = 64

(x_train, _), (x_test, _) = load_data()

all_images = np.concatenate([x_train, x_test])

all_images = all_images.astype("float32") / 255

all_images = np.reshape(all_images, (-1, 28, 28, 1))

dataset = tf.data.Dataset.from_tensor_slices(all_images)

dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)
def define_discriminator():

    model = tf.keras.Sequential(

        [

            layers.Conv2D(32, 3, strides=2, padding='same',

                          input_shape=(28, 28, 1)),

            layers.LeakyReLU(alpha=0.2),

            layers.Dropout(0.5),

            

            layers.Conv2D(64, 3, padding='same'),

            layers.BatchNormalization(),

            layers.LeakyReLU(alpha=0.2),

            layers.Dropout(0.5),

            

            layers.Conv2D(128, 3, strides=2, padding='same'),

            layers.BatchNormalization(),

            layers.LeakyReLU(alpha=0.2),

            layers.Dropout(0.5),

            

            layers.Conv2D(256, 3, padding='same'),

            layers.BatchNormalization(),

            layers.LeakyReLU(alpha=0.2),

            layers.Dropout(0.5),

            

            layers.GlobalMaxPooling2D(),

            layers.Dense(1, activation='sigmoid')

        ]

    )

    

    return model
if TPU_used:

    with tpu_strategy.scope():

        discriminator = define_discriminator()

else:

    discriminator = define_discriminator()

discriminator.summary()
def define_generator(latent_size):

    model = tf.keras.Sequential(

        [

            layers.Dense(7 * 7 * 128, input_dim=latent_size),

            layers.LeakyReLU(alpha=0.2),

            layers.Reshape((7, 7, 128)),

            

            layers.Conv2DTranspose(128, 4, strides=2, padding='same',

                                   kernel_initializer='glorot_normal'),

            layers.LeakyReLU(alpha=0.2),

            layers.BatchNormalization(),

            

            layers.Conv2DTranspose(128, 4, strides=2, padding='same',

                                   kernel_initializer='glorot_normal'),

            layers.LeakyReLU(alpha=0.2),

            layers.BatchNormalization(),

            

            layers.Conv2D(1, 7, padding='same',

                          activation='tanh',

                          kernel_initializer='glorot_normal')

        ]

    )

    

    return model
if TPU_used:

    with tpu_strategy.scope():

        generator = define_generator(latent_dim)

else:

    generator = define_generator(latent_dim)

generator.summary()
class GAN(tf.keras.Model):

    def __init__(self, discriminator, generator, latent_dim):

        super(GAN, self).__init__()

        self.discriminator = discriminator

        self.generator = generator

        self.latent_dim = latent_dim



    def compile(self, d_optimizer, g_optimizer, loss_fn):

        super(GAN, self).compile()

        self.d_optimizer = d_optimizer

        self.g_optimizer = g_optimizer

        self.loss_fn = loss_fn



    def train_step(self, real_images):

        if isinstance(real_images, tuple):

            real_images = real_images[0]

        # Sample random points in the latent space

        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))



        # Decode them to fake images

        generated_images = self.generator(random_latent_vectors)



        # Combine them with real images

        combined_images = tf.concat([generated_images, real_images], axis=0)



        # Assemble labels discriminating real from fake images

        labels = tf.concat(

            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0

        )

        # Add random noise to the labels - important trick!

        labels += 0.05 * tf.random.uniform(tf.shape(labels))



        # Train the discriminator

        with tf.GradientTape() as tape:

            predictions = self.discriminator(combined_images)

            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

        self.d_optimizer.apply_gradients(

            zip(grads, self.discriminator.trainable_weights)

        )



        # Sample random points in the latent space

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))



        # Assemble labels that say "all real images"

        misleading_labels = tf.zeros((batch_size, 1))



        # Train the generator (note that we should *not* update the weights

        # of the discriminator)!

        with tf.GradientTape() as tape:

            predictions = self.discriminator(self.generator(random_latent_vectors))

            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)

        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}
class GANMonitor(tf.keras.callbacks.Callback):

    def __init__(self, num_img=3, latent_dim=128):

        self.num_img = num_img

        self.latent_dim = latent_dim



    def on_epoch_end(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))

        generated_images = self.model.generator(random_latent_vectors)

        generated_images *= 255

        generated_images.numpy()

        for i in range(self.num_img):

            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])

            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
if TPU_used:

    with tpu_strategy.scope():

        gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)

        gan.compile(

            d_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),

            g_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),

            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True,

                                                       reduction=tf.keras.losses.Reduction.NONE),

        )

else:

    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)

    gan.compile(

        d_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),

        g_optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),

        loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),

    )

gan.fit(

    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim)]

)
!ls
Image.open("generated_img_2_20.png")