# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install tensorflow-gpu==2.0.0beta1
import tensorflow as tf

import pathlib

from PIL import Image

import numpy as np

from tensorflow.python.keras import layers

from tensorflow.python import keras

import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

import glob

from numpy import random

import zipfile

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.__version__
image = glob.glob('../input/all-dogs/all-dogs/*')

breed = glob.glob('../input/annotation/Annotation/*')

annot = glob.glob('../input/annotation/Annotation/*/*')

print(len(image), len(breed), len(annot))
def get_bbox(annot):

    """

    This extracts and returns values of bounding boxes

    """

    xml = annot

    tree = ET.parse(xml)

    root = tree.getroot()

    objects = root.findall('object')

    bbox = []

    for o in objects:

        bndbox = o.find('bndbox')

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        bbox.append((xmin,ymin,xmax,ymax))

    return bbox
def get_image(annot):

    """

    Retrieve the corresponding image given annotation file

    """

    img_path = '../input/all-dogs/all-dogs/'

    file = annot.split('/')

    img_filename = img_path+file[-1]+'.jpg'

    return img_filename
n_x = 64

n_c = 3

dogs = np.zeros((len(image), n_x, n_x, n_c))

print(dogs.shape)
for a in range(len(image)):

    bbox = get_bbox(annot[a])

    dog = get_image(annot[a])

    if dog == '../input/all-dogs/all-dogs/n02105855_2933.jpg':   # this jpg is not in the dataset

        continue

    im = Image.open(dog)

    im = im.crop(bbox[0])

    im = im.resize((64,64), Image.ANTIALIAS)

    dogs[a,:,:,:] = np.asarray(im) / 255. * 2 - 1
plt.figure(figsize=(15,8))

n_images = 60

select = random.randint(low=0,high=dogs.shape[0],size=n_images)

for i, index in enumerate(select):  

    plt.subplot(6, 10, i+1)

    plt.imshow(dogs[index])

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
batch_size = 128

seed_dim = 100

seed = tf.random.normal([batch_size, seed_dim])
dogs = tf.cast(dogs, tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(dogs).batch(batch_size=batch_size).shuffle(len(image))

sample = next(iter(dataset))

sample.shape
# build model

class Generator(keras.Model):

    def __init__(self):

        super().__init__()

        self.fc1 = layers.Dense(4*4*512)

        self.dconv1 = layers.Conv2DTranspose(256, 5, 2, padding='same')

        self.bn1 = layers.BatchNormalization()

        self.dconv2 = layers.Conv2DTranspose(128, 5, 2, padding='same')

        self.bn2 = layers.BatchNormalization()

        self.dconv3 = layers.Conv2DTranspose(64, 5, 2, padding='same')

        self.bn3 = layers.BatchNormalization()

        self.dconv4 = layers.Conv2DTranspose(3, 5, 2, padding='same')

    def call(self, inputs, training=None):

        x = self.fc1(inputs)

        x = tf.reshape(x, [-1, 4, 4, 512])

        x = tf.nn.leaky_relu(x)

        x = self.dconv1(x)

        x = self.bn1(x)

        x = tf.nn.leaky_relu(x)

        x = self.dconv2(x)

        x = self.bn2(x)

        x = tf.nn.leaky_relu(x)

        x = self.dconv3(x)

        x = self.bn3(x)

        x = tf.nn.leaky_relu(x)

        x = self.dconv4(x)

        return x
class Discriminator(keras.Model):

    def __init__(self):

        super().__init__()

        self.conv1 = layers.Conv2D(64, 5, 2, padding='same')

        self.dp1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv2D(128, 5, 2, padding='same')

        self.dp2 = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(256, 5, 2, padding='same')

        self.flat = layers.Flatten()

        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):

        x = self.conv1(inputs)

        x = tf.nn.leaky_relu(x)

        x = self.dp1(x)

        x = self.conv2(x)

        x = tf.nn.leaky_relu(x)

        x = self.dp2(x)

        x = self.conv3(x)

        x = tf.nn.leaky_relu(x)

        x = self.flat(x)

        x = self.fc(x)

        return x
# define losses

def disc_loss(real_logits, fake_logits):

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits), logits=real_logits))

    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits))

    return real_loss + fake_loss



def gen_loss(fake_logits):

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits), logits=fake_logits))
generator = Generator()

discriminator = Discriminator()
gen_optimizer = tf.optimizers.Adam(0.0001)

disc_optimizer = tf.optimizers.Adam(0.0001)
# train

def train_step(imgs):

    noise = tf.random.normal([imgs.shape[0], seed_dim])

    

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        fake_img = generator(noise)

        fake_logits = discriminator(fake_img, training=True)

        real_logits = discriminator(imgs, training=True)

        

        g_loss = gen_loss(fake_logits)

        d_loss = disc_loss(real_logits, fake_logits)

    

    gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)

    disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
def generate_and_save_imgs(generator, seed, epoch):

    pred = generator(seed)

    for i in range(6):

        plt.subplot(2,3,i+1)

        plt.imshow(pred[i] * 127.5 + 127.5)

        plt.axis('off')

    plt.show()
def train(dataset, epoches):

    for epoch in range(epoches):

        for imgs in dataset:

            train_step(imgs)

        if (epoch + 1) % 50 == 0:

            generate_and_save_imgs(generator, seed, epoch)
train(dataset, epoches = 500)
# save output

z = zipfile.PyZipFile('images.zip', mode='w')

for d in range(10000):

    img = generator(tf.random.normal([1, 100]))

    img = tf.squeeze(img, axis=0)

    img = img * 127.5 + 127.5

    img = img.numpy()

    img = Image.fromarray(img.astype('uint8'),'RGB')

    f = str(d)+'.png'

    img.save(f,'PNG')

    z.write(f)

    os.remove(f)

z.close()