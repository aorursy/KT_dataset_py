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
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

from keras.datasets import mnist
img_rows = 128
img_cols = 128
channels = 3
num_classes = 10
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
def build_disk_and_q_net(img_shape=img_shape, num_classes=num_classes):

  img = Input(shape=img_shape)

  # Shared layers between discriminator and recognition network
  model = Sequential()
  model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.5))
  model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
  model.add(ZeroPadding2D(padding=((0,1),(0,1))))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.5))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.5))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.5))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.5))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(16, kernel_size=3, strides=1, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.5))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Flatten())

  img_embedding = model(img)

  # Discriminator
  validity = Dense(1, activation='sigmoid')(img_embedding)

  # Recognition
  q_net = Dense(8, activation='tanh')(img_embedding)
  #######label = Dense(num_classes, activation='softmax')(q_net)

  # Return discriminator and recognition network
  return Model(img, validity), Model(img, label)

def build_generator(latent_dim=latent_dim, channels=channels):

  model = Sequential()

  model.add(Dense(256 * 6 * 4, activation="relu", input_dim=latent_dim))
  model.add(Reshape((4, 6, 256)))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(128, kernel_size=3, strides = 1, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Dropout(0.6))
  model.add(Conv2D(64, kernel_size=3, strides = 1, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Dropout(0.6))
  model.add(Conv2D(32, kernel_size=3, strides = 1, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Dropout(0.6))
  model.add(Conv2D(16, kernel_size=3, strides = 1, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Dropout(0.6))    
  model.add(Conv2D(channels, kernel_size=3, strides = 1, padding='same'))
  model.add(Activation("tanh"))
  model.add(Dropout(0.6))  

  ########gen_input = Input(shape=(latent_dim,))
  img = model(gen_input)

  model.summary()

  return Model(gen_input, img)
def mutual_info_loss(c, c_given_x):
  """The mutual information metric we aim to minimize"""
  eps = 1e-8
  conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
  entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

  return conditional_entropy + entropy
def sample_generator_input(batch_size, num_classes=num_classes):
  # Generator inputs
  sampled_noise = np.random.normal(0, 1, (batch_size, 62))
  sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
  sampled_labels = to_categorical(sampled_labels, num_classes=num_classes)

  return sampled_noise, sampled_labels
def train(epochs, batch_size=128, sample_interval=50):

  # Load the dataset
  (X_train, y_train), (_, _) = mnist.load_data()

  # Rescale -1 to 1
  X_train = (X_train.astype(np.float32) - 127.5) / 127.5
  X_train = np.expand_dims(X_train, axis=3)
  y_train = y_train.reshape(-1, 1)

  # Adversarial ground truths
  valid = np.ones((batch_size, 1))
  fake = np.zeros((batch_size, 1))

  for epoch in tqdm(range(epochs)):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    # Sample noise and categorical labels
    sampled_noise, sampled_labels = sample_generator_input(batch_size)
    gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)

    # Generate a half batch of new images
    gen_imgs = generator.predict(gen_input)

    # Train on real and generated data
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

    # Avg. loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator and Q-network
    # ---------------------

    g_loss = combined.train_on_batch(gen_input, [valid, sampled_labels])

    # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        sample_images(epoch)

def sample_images(epoch, num_classes=num_classes):
  r, c = 10, 10

  fig, axs = plt.subplots(r, c)
  for i in range(c):
      sampled_noise, _ = sample_generator_input(c)
      label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=num_classes)
      gen_input = np.concatenate((sampled_noise, label), axis=1)
      gen_imgs = generator.predict(gen_input)
      gen_imgs = 0.5 * gen_imgs + 0.5
      for j in range(r):
          axs[j,i].imshow(gen_imgs[j,:,:,0],cmap='pink')
          axs[j,i].axis('off')
        
  fig.savefig("%d.jpeg" % epoch)
  plt.close()
optimizer = Adam(0.0002, 0.5)
losses = ['binary_crossentropy',mutual_info_loss]

# Build and the discriminator and recognition network
discriminator, auxilliary = build_disk_and_q_net()

discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

# Build and compile the recognition network Q
auxilliary.compile(loss=[mutual_info_loss], optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = build_generator()

# The generator takes noise and the target label as input
# and generates the corresponding digit of that label
gen_input = Input(shape=(latent_dim,))
img = generator(gen_input)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated image as input and determines validity
valid = discriminator(img)
# The recognition network produces the label
target_label = auxilliary(img)

# The combined model  (stacked generator and discriminator)
combined = Model(gen_input, [valid, target_label])
combined.compile(loss=losses, optimizer=optimizer)

train(epochs=2000, batch_size=128, sample_interval=50)
sample_images(2, num_classes=num_classes)
