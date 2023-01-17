!pip install git+https://github.com/tensorflow/examples.git
#IMPORTS

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import os, time

from kaggle_datasets import KaggleDatasets

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
print(tf.__version__)
# Configuration

class Configuration:

    """Class containing most of the parameters or hyperparameters used

    throughout the notebook."""

    

    epochs = 30

    MONET_TFREC = "/monet_tfrec/*.tfrec"

    MONET_JPG = "/monet_jpg/*.jpg"

    PHOTO_TFREC = "/photo_tfrec/*.tfrec"

    PHOTO_JPG = "/photo_jpg/*.jpg"

    BATCH_SIZE = 8

    IMAGE_SIZE = [256, 256]

    BUFFER = 10000

    steps_per_epoch = 0

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)



cfg = Configuration()
# Setting up the TPU

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)



AUTOTUNE = tf.data.experimental.AUTOTUNE
monet_jpg = tf.io.gfile.glob("../input/gan-getting-started/monet_jpg/*.jpg")

cfg.steps_per_epoch = len(monet_jpg)
class MonetDataset:

        def __init__(self,config):

            """Creates a data of TFRecord files."""

            self.cfg = config

            gcs_path = KaggleDatasets().get_gcs_path()

            self.monet_files = tf.io.gfile.glob(gcs_path+self.cfg.MONET_TFREC)

            self.photo_files = tf.io.gfile.glob(gcs_path + self.cfg.PHOTO_TFREC)

            

        def decode_image(self, image):

            """Function to preprocess the image prior to training."""

            img = tf.image.decode_jpeg(image, channels=3)

            img = tf.cast(img, tf.float32)

            img = img/127.5 - 1

            img = tf.reshape(img, [*self.cfg.IMAGE_SIZE, 3])

            return img

        

        def read_tfrecord(self, instance):

            """Function to extract data from TFRecordDataset Instance."""

            tfrecordformat = {

                    "image_name": tf.io.FixedLenFeature([], tf.string),

                    "image": tf.io.FixedLenFeature([], tf.string),

                    "target": tf.io.FixedLenFeature([], tf.string)

                   }

            example = tf.io.parse_single_example(instance, tfrecordformat)

            return self.decode_image(example["image"])

        

        def prepare_dataset(self, monet=True):

            """Main function to prepare the input pipeline.

            Args: 

            monet- bool value

            Determines if we wanna generate monet dataset or the photo dataset"""

            dataset = tf.data.TFRecordDataset(self.monet_files if monet else self.photo_files, num_parallel_reads=AUTOTUNE)

            dataset = dataset.map(self.read_tfrecord, num_parallel_calls=AUTOTUNE)

            dataset = dataset.map(self.random_jitter, num_parallel_calls=AUTOTUNE)

            dataset = dataset.repeat()

            dataset = dataset.shuffle(self.cfg.BUFFER)

            dataset = dataset.batch(self.cfg.BATCH_SIZE)

            dataset = dataset.prefetch(AUTOTUNE)

            return dataset



        def random_crop(self, image):

            """Function to perform random cropping."""

            image = tf.image.random_crop(image, [*self.cfg.IMAGE_SIZE, 3])

            return image

        

        def random_jitter(self, image):

            """Function to perform random jittering."""

            image = tf.image.resize(image, [286, 286])

            image = self.random_crop(image)

            

            if tf.random.uniform([], 0, 1) > 0.5:

                image = tf.image.random_flip_left_right(image)

            return image

        

        def visualize_data(self, data):

            """Utility function to visualize the samples in the dataset instance being provided."""

            fig, ax = plt.subplots(2, self.cfg.BATCH_SIZE//2, figsize=(16, 4)) # Figsize->W x H

            ax = ax.flatten()

            for i, im in zip(range(self.cfg.BATCH_SIZE), data):

                im = im*0.5 + 0.5

                ax[i].imshow(im)

                ax[i].axis("off")

            plt.show()
# Creating instance of dataset

dataset = MonetDataset(Configuration())
# Creating seperate monet and photo dataset.

monet_dataset = dataset.prepare_dataset(monet=True)

photo_dataset = dataset.prepare_dataset(monet=False)
example = next(iter(monet_dataset))

dataset.visualize_data(example)
dataset.visualize_data(next(iter(photo_dataset)))
class CyclicGAN(tf.keras.Model):

    """Class to build and train custom CyclicGAN architecture."""

    def __init__(self, 

                monet_generator, 

                photo_generator,

                monet_discriminator,

                photo_discriminator,

                lambda_cyclic = 10):

        super(CyclicGAN, self).__init__()

        self.m_gen = monet_generator

        self.p_gen = photo_generator

        self.m_disc = monet_discriminator

        self.p_disc = photo_discriminator

        self.lambda_ = lambda_cyclic

    

    def compile(self,

                m_gen_optimizer,

                p_gen_optimizer,

                m_disc_optimizer,

                p_disc_optimizer,

                gen_loss,

                disc_loss,

                cyclic_loss,

                identity_loss

               ):

        """Function to set the optimizers and metrics used for the model training."""

        super(CyclicGAN, self).compile()

        self.m_gen_optimizer= m_gen_optimizer

        self.p_gen_optimizer = p_gen_optimizer

        self.m_disc_optimizer = m_disc_optimizer

        self.p_disc_optimizer = p_disc_optimizer

        self.gen_loss = gen_loss

        self.disc_loss = disc_loss

        self.cyclic_loss = cyclic_loss

        self.identity_loss = identity_loss

    

    def train_step(self, batch_data):

        """Function to run a single step of training."""

        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:

            

            # Getting Generator and Discriminator output. 

            fake_monet = self.m_gen(real_photo, training=True)

            cycled_photo = self.p_gen(fake_monet, training=True)

            

            fake_photo = self.p_gen(real_monet, training=True)

            cycled_monet = self.m_gen(fake_photo, training=True)

            

            same_monet = self.m_gen(real_monet, training=True)

            same_photo = self.p_gen(real_photo, training=True)

            

            disc_real_monet = self.m_disc(real_monet, training=True)

            disc_fake_monet = self.m_disc(fake_monet, training=True)

            

            disc_real_photo = self.p_disc(real_photo, training=True)

            disc_fake_photo = self.p_disc(fake_photo, training=True)

            

            # Calculate Losses

            cycle_loss = self.lambda_*(self.cyclic_loss(real_monet, cycled_monet)+self.cyclic_loss(real_photo, cycled_photo))

            identity_loss = self.lambda_ * (self.identity_loss(real_monet, same_monet) + self.identity_loss(real_photo, same_photo))

            

            monet_gen_loss = self.gen_loss(disc_fake_monet)

            photo_gen_loss = self.gen_loss(disc_fake_photo)

            

            total_monet_gen_loss = monet_gen_loss + cycle_loss + identity_loss

            total_photo_gen_loss = photo_gen_loss + cycle_loss + identity_loss

            

            monet_disc_loss = self.disc_loss(disc_real_monet, disc_fake_monet)

            photo_disc_loss = self.disc_loss(disc_real_photo, disc_fake_photo)

            

            # Calculate Gradients

            monet_gen_gradient = tape.gradient(total_monet_gen_loss, self.m_gen.trainable_variables)

            photo_gen_gradient = tape.gradient(total_photo_gen_loss, self.p_gen.trainable_variables)

            

            monet_disc_gradient = tape.gradient(monet_disc_loss, self.m_disc.trainable_variables)

            photo_disc_gradient = tape.gradient(photo_disc_loss, self.p_disc.trainable_variables)

            

            self.m_gen_optimizer.apply_gradients(zip(monet_gen_gradient, self.m_gen.trainable_variables))

            self.p_gen_optimizer.apply_gradients(zip(photo_gen_gradient, self.p_gen.trainable_variables))

            

            # Apply Gradients

            self.m_disc_optimizer.apply_gradients(zip(monet_disc_gradient, self.m_disc.trainable_variables))

            self.p_disc_optimizer.apply_gradients(zip(photo_disc_gradient, self.p_disc.trainable_variables))

            

            return {

                "monet_generator_loss": total_monet_gen_loss,

                "monet_discriminator_loss": monet_disc_loss,

                "photo_generator_loss": total_photo_gen_loss,

                "photo_discriminator_loss": photo_disc_loss

            }
OUTPUT_CHANNELS = 3

with strategy.scope():

    monet_gen = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type="instancenorm")

    photo_gen = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type="instancenorm")



    monet_disc = pix2pix.discriminator(norm_type="instancenorm", target=False)

    photo_disc = pix2pix.discriminator(norm_type="instancenorm", target=False)
with strategy.scope():

    model = CyclicGAN(monet_gen, photo_gen, monet_disc, photo_disc)

    

    # Prepairing the loss functions.

    def generator_loss(generated_op):

        return cfg.loss(tf.ones_like(generated_op), generated_op)

    

    def discriminator_loss(disc_real_op, disc_fake_op):

        real_loss = cfg.loss(tf.ones_like(disc_real_op), disc_real_op)

        fake_loss = cfg.loss(tf.zeros_like(disc_fake_op), disc_fake_op)

        total_loss = real_loss + fake_loss

        return total_loss * 0.5

    

    def cyclic_loss(real_image, cycled_image):

        return tf.reduce_mean(tf.abs(real_image - cycled_image))

    

    def identity_loss(real_image, same_image):

        return tf.reduce_mean(tf.abs(real_image-same_image))

    

    m_gen_optim = tf.keras.optimizers.Adam(lr=2e-04, beta_1=0.5)

    p_gen_optim = tf.keras.optimizers.Adam(lr=2e-04, beta_1=0.5)

    m_disc_optim = tf.keras.optimizers.Adam(lr=2e-04, beta_1=0.5)

    p_disc_optim = tf.keras.optimizers.Adam(lr=2e-04, beta_1=0.5)
with strategy.scope():

    model.compile(m_gen_optim,

                 p_gen_optim,

                 m_disc_optim,

                 p_disc_optim,

                 generator_loss,

                 discriminator_loss,

                 cyclic_loss,

                 identity_loss)
# Visualizing Generator model.

tf.keras.utils.plot_model(monet_gen, show_shapes=True, dpi=64)
# Visualizing Discriminator model

tf.keras.utils.plot_model(monet_disc, show_shapes=True, dpi=96)
with strategy.scope():

    model.fit(tf.data.Dataset.zip((monet_dataset, photo_dataset)), epochs=cfg.epochs,

             steps_per_epoch = cfg.steps_per_epoch)
photo_example = next(iter(photo_dataset))

predict_img = monet_gen.predict(tf.expand_dims(photo_example[0], axis=0))
fig, ax = plt.subplots(1, 2, figsize=(8,8))

ax = ax.flatten()

ax[0].imshow(photo_example[0]*0.5 + 0.5)

out  = (predict_img[0]*127.5 + 127.5).astype(np.uint8)

ax[1].imshow(out)
import PIL

!mkdir ../images
photo_jpg = tf.io.gfile.glob("../input/gan-getting-started/photo_jpg/*.jpg")
for i, image in zip(range(1, len(photo_jpg)+1), photo_dataset):

    prediction = monet_gen(image, training=False)[0].numpy()

    prediction = (prediction*127.5 + 127.5).astype(np.uint8)

    im = PIL.Image.fromarray(prediction)

    im.save(f"../images/{i}.jpg")

    if(i%100==0):

        print(f"Processed {i} images")
import shutil

shutil.make_archive("/kaggle/working/images", "zip", "/kaggle/images")