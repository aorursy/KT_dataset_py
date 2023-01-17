# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install --upgrade pip
!pip install git+https://github.com/tensorflow/docs
import glob
import imageio
import time 
import keras
import PIL
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt

from keras import Sequential
from keras.optimizers import Adam
import tensorflow_docs.vis.embed as embed
from keras.losses import BinaryCrossentropy

(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images.reshape(-1, 28, 28,1)
train_images = (train_images - 127.5) / 127.5
images = train_images
train_images.shape ,train_labels.shape
learning_rate = 0.0001
batch_size = 256
epochs = 500
break_num = 5
image_dim = train_images[0].shape
random_noise = 100
image_dim
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, BatchNormalization, Reshape, Conv2DTranspose, Conv1D
def discriminator():
    model = Sequential(name = "discriminator")
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=image_dim))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))
    
    model.summary()

    return model
def generator():
    model = Sequential(name="generator")
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))    # adding noise
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    model.summary()
    return model
gen = generator()

noise = tf.random.normal([1, 100])
generated_image = gen(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])
disc = discriminator()
print(disc(generated_image))
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    return fake_loss + real_loss
def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss
discriminator_optimizer = Adam(learning_rate)

generator_optimizer = Adam(learning_rate)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=gen,
                                 discriminator=disc)
def train_step(input_image):
    noise = tf.random.normal([batch_size, random_noise])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise)

        real_output = disc(input_image)
        fake_output = disc(generated_images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    g_gen = gen_tape.gradient(gen_loss, gen.trainable_variables)
    g_disc = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(g_gen, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(g_disc, disc.trainable_variables))
    return gen_loss, disc_loss
noise_gan = tf.random.normal([batch_size, random_noise])
def save_output(model, epoch):
    prediction = model(noise_gan)

    fig = plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(prediction[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
train_images = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)

for epoch in range(epochs):
    start = time.time()

    for x_batch in train_images:
        loss = train_step(x_batch)
        
    if (epoch + 1) % break_num == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
    if epoch %break_num ==0 or epoch==epochs-1 and epoch!=0:
        print ('Time for epoch {} is {} sec and Generator Loss :{} and Discriminator loss :{}'.format(epoch + 1, time.time()-start, loss[0], loss[1]))
        save_output(gen, epoch+1)

def display_image(epoch_no, file):
    return PIL.Image.open(file.format(epoch_no))
display_image(epoch+1, 'image_at_epoch_{:04d}.png')
def create_gif(file, collection):

    with imageio.get_writer(file, mode='I') as writer:
        filenames = glob.glob(collection+'*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    return file
file  = create_gif("result.gif", "image")
embed.embed_file(file)
from keras.utils import to_categorical

train_images = images
labels = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, labels)

learning_rate = 0.0001
batch_size = 64

random_noise = 100
def discriminator():
    model = Sequential(name = "discriminator")
    model.add(Dense(7*7*256, use_bias=False, input_shape=(794,)))
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))
    
    return model

def generator():
    model = Sequential(name="generator")
    model.add(Dense(7*7*256, use_bias=False, input_shape=(110,)))    # adding noise
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    model.add(Reshape((784,)))
    
    return model

print(generator().summary())
print(discriminator().summary())
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    return fake_loss + real_loss
def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


discriminator_optimizer = Adam(learning_rate)

generator_optimizer = Adam(learning_rate)
gen = generator()
disc = discriminator()

def train_step(input_image, labels):
    noise = tf.random.normal([batch_size, random_noise])
    noise = tf.concat(axis=1, values=[noise, np.stack(labels)])
    
    input_image = tf.concat(axis=1, values = [tf.cast(input_image, tf.float32), labels])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise)
        generated_images = tf.concat(axis=1, values=[generated_images, labels])
        real_output = disc(input_image)
        fake_output = disc(generated_images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    g_gen = gen_tape.gradient(gen_loss, gen.trainable_variables)
    g_disc = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(g_gen, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(g_disc, disc.trainable_variables))
    return gen_loss,disc_loss
    
noise = tf.random.normal([batch_size, random_noise])
Y_label = np.zeros(shape = [batch_size, 10])
Y_label[:, 4] = 1    # For Coat
noise_gan = tf.concat(axis=1, values=[noise, Y_label])

def save_output(model, epoch):
    
    prediction = model(noise_gan)
    prediction = prediction.numpy().reshape(prediction.shape[0],28, 28, 1)
    
    fig = plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(prediction[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')
    plt.savefig('conditional_image_at_epoch_{:04d}.png'.format(epoch))
    
train_labels = tf.data.Dataset.from_tensor_slices(train_labels[:59968]).shuffle(128).batch(batch_size)
train_images = tf.data.Dataset.from_tensor_slices(train_images.reshape(train_images.shape[0], 784)[:59968]).shuffle(128).batch(batch_size)

for epoch in range(epochs):
    start = time.time()

    for x_batch, labels in zip(train_images, train_labels):
        loss = train_step(x_batch, labels)
        
    if (epoch + 1) % break_num == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
        
    if epoch %break_num ==0 or epoch==epochs-1 and epoch!=0:
        print ('Time for epoch {} is {} sec and Generator Loss :{} and Discriminator loss :{}'.format(epoch + 1, time.time()-start, loss[0], loss[1]))
        save_output(gen, epoch+1)
display_image(epoch+1, 'conditional_image_at_epoch_{:04d}.png')
display.clear_output(wait=True)

noise = tf.random.normal([batch_size, random_noise])
feature_map = { "t-shirt":0,
                 "trouser":1,
                 "pullover":2,
                 "dress":3,
                 "coat":4,
                 "sandal":5,
                 "sirt":6,
                 "sneaker":7,
                 "bag":8,
                 "ankle boot": 9
                }
inp = "coat"

Y_label = np.zeros(shape = [batch_size, 10])
Y_label[:, feature_map[inp]] = 1
noise = tf.concat(axis=1, values=[noise, Y_label])

prediction = gen(noise)
prediction = prediction.numpy().reshape(prediction.shape[0],28, 28, 1)
fig = plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(prediction[i, :, :, 0] * 127.5 + 127.5)
    plt.axis('off')
plt.show()
file = create_gif("result.gif", "conditional_image")
embed.embed_file(file)
