! pip install -q git+https://github.com/tensorflow/examples.git
# Imports and TPU configs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import PIL
import shutil

import tensorflow as tf
import tensorflow.keras
import tensorflow_addons as tfa
from tensorflow_examples.models.pix2pix import pix2pix

from kaggle_datasets import KaggleDatasets
from IPython.display import clear_output

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Device: {tpu.master()}")
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print(f"Number of replicas: {strategy.num_replicas_in_sync}")
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)
# Get Google Cloud Storage path and data
GCS_PATH = KaggleDatasets().get_gcs_path()

MONET_FL = tf.io.gfile.glob(str(GCS_PATH+'/monet_tfrec/*.tfrec'))
print(f"Monet TFRecord Files: {len(MONET_FL)}")

PHOTO_FL = tf.io.gfile.glob(str(GCS_PATH+'/photo_tfrec/*.tfrec'))
print(f"Photo TFRecord Files: {len(PHOTO_FL)}")
IMAGE_SIZE = [256, 256]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image
def load_dataset(filenames, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset
monet_ds = load_dataset(MONET_FL, labeled=True).batch(1)
photo_ds = load_dataset(PHOTO_FL, labeled=True).batch(1)
ex_monet = next(iter(monet_ds))
ex_photo = next(iter(photo_ds))

plt.subplot(121)
plt.title("Monet Image")
plt.imshow(ex_monet[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title("Normal Image")
plt.imshow(ex_photo[0] * 0.5 + 0.5)
# Define the generator and discriminator based on pix2pix generator
gen_g = pix2pix.unet_generator(3, norm_type='instancenorm')
gen_f = pix2pix.unet_generator(3, norm_type='instancenorm')

dis_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
dis_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
# Loss functions
LAMBDA = 10
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def dis_loss(real, generated):
    real_loss = loss_fn(tf.ones_like(real), real)
    generated_loss = loss_fn(tf.zeros_like(generated), generated)
    
    disc_loss = real_loss + generated_loss
    
    return disc_loss * 0.5

def gen_loss(generated):
    return loss_fn(tf.ones_like(generated), generated)

def cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return loss * LAMBDA

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return 0.5 * LAMBDA * loss
generator_g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_path = "/kaggle/working/checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=gen_g,
                           generator_f=gen_f,
                           discriminator_x=dis_x,
                           discriminator_y=dis_y,
                           generator_g_optimizer=generator_g_opt,
                           generator_f_optimizer=generator_f_opt,
                           discriminator_x_optimizer=discriminator_x_opt,
                           discriminator_y_optimizer=discriminator_y_opt)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
def generate_image(model, noise):
    prediction = model(noise)
    plt.figure(figsize=(12, 12))
    display_list = [noise[0], prediction[0]]
    title=['Input Noise', 'Generated Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
@tf.function
def fit_one_epoch(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = gen_g(real_x, training=True)
        cycled_x = gen_f(fake_y, training=True)
        
        fake_x = gen_g(real_y, training=True)
        cycled_y = gen_f(fake_x, training=True)
        
        same_x = gen_f(real_x, training=True)
        same_y = gen_g(real_y, training=True)
        
        dis_real_x = dis_x(real_x, training=True)
        dis_real_y = dis_y(real_y, training=True)
        
        dis_fake_x = dis_x(fake_x, training=True)
        dis_fake_y = dis_y(fake_y, training=True)
        
        # Calculate Loss
        gen_g_loss = gen_loss(dis_fake_y)
        gen_f_loss = gen_loss(dis_fake_x)
        
        cy_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)
        
        total_gen_g_loss = gen_g_loss + cy_loss + identity_loss(real_x, same_y)
        total_gen_f_loss = gen_f_loss + cy_loss + identity_loss(real_x, same_x)
        
        disc_x_loss = dis_loss(dis_real_x, dis_fake_y)
        disc_y_loss = dis_loss(dis_real_y, dis_fake_y)
        
    # Gradient stuff
    gen_g_grads = tape.gradient(total_gen_g_loss, gen_g.trainable_variables)
    gen_f_grads = tape.gradient(total_gen_f_loss, gen_f.trainable_variables)

    dis_x_grads = tape.gradient(disc_x_loss, dis_x.trainable_variables)
    dis_y_grads = tape.gradient(disc_y_loss, dis_y.trainable_variables)

    # Apply gradients to optimizer
    generator_g_opt.apply_gradients(zip(gen_g_grads, gen_g.trainable_variables))
    generator_f_opt.apply_gradients(zip(gen_f_grads, gen_f.trainable_variables))

    discriminator_x_opt.apply_gradients(zip(dis_x_grads, dis_x.trainable_variables))
    discriminator_y_opt.apply_gradients(zip(dis_y_grads, dis_y.trainable_variables))
# Training loop
nb_epochs = 80

for epoch in range(nb_epochs):
    n = 0
    for image_x, image_y in tf.data.Dataset.zip((monet_ds, photo_ds)):
        fit_one_epoch(image_x, image_y)
        if n % 10 == 0:
            print('.', end=' ')
        n += 1
    clear_output(wait=True)
    generate_image(gen_g, ex_monet)
! mkdir "/kaggle/working/images"

i = 1
for img in photo_ds:
    prediction = gen_g(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    im = PIL.Image.fromarray(prediction)
    im.save("/kaggle/working/images" + str(i) + ".jpg")
    i += 1
shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")
