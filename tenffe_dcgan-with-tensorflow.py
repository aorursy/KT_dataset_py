# -*- coding: utf-8 -*-



import tensorflow as tf

import numpy as np

import urllib

import tarfile

import os

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

%matplotlib inline

from imageio import imread, imsave, mimsave

from PIL import Image

import glob

import shutil
import numpy as np # linear algebra

import xml.etree.ElementTree as ET # for parsing XML

import matplotlib.pyplot as plt # to show images

from PIL import Image # to read images

import os

import glob



root_images="../input/all-dogs/all-dogs/"

root_annots="../input/annotation/Annotation/"
all_images=os.listdir("../input/all-dogs/all-dogs/")

print(f"Total images : {len(all_images)}")



breeds = glob.glob('../input/annotation/Annotation/*')

annotation=[]

for b in breeds:

    annotation+=glob.glob(b+"/*")

print(f"Total annotation : {len(annotation)}")



breed_map={}

for annot in annotation:

    breed=annot.split("/")[-2]

    index=breed.split("-")[0]

    breed_map.setdefault(index,breed)

    

print(f"Total Breeds : {len(breed_map)}")
def bounding_box(image):

    bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])

    tree = ET.parse(bpath)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox') # reading bound box

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        

    return (xmin,ymin,xmax,ymax)



def get_crop_image(image):

    bbox=bounding_box(image)

    im=Image.open(os.path.join(root_images,image))

    im=im.crop(bbox)

    return im
%%time

plt.figure(figsize=(10,10))

for i,image in enumerate(all_images):

    im=get_crop_image(image)

    

    plt.subplot(3,3,i+1)

    plt.axis("off")

    plt.imshow(im)    

    if(i==8):

        break
%%time

path = '../input/all-dogs'

dataset = 'all-dogs'

data_path = os.path.join(path, dataset)

images = glob.glob(os.path.join(data_path, '*.*')) 

print(len(images))
images[0]
z_dim = 1000



WIDTH = 64

HEIGHT = 64



OUTPUT_DIR = 'samples_dogs'

if not os.path.exists(OUTPUT_DIR):

    os.mkdir(OUTPUT_DIR)



GEN_DIR = 'generated_dogs'

if not os.path.exists(GEN_DIR):

    os.mkdir(GEN_DIR)

    

X = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='X')

noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')

is_training = tf.placeholder(dtype=tf.bool, name='is_training')



def lrelu(x, leak=0.2):

    return tf.maximum(x, leak * x)



def sigmoid_cross_entropy_with_logits(x, y):

    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
def discriminator(image, reuse=None, is_training=is_training):

    momentum = 0.9

    with tf.variable_scope('discriminator', reuse=reuse):

        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))

        

        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')

        h1 = lrelu(tf.layers.batch_normalization(h1, training=is_training, momentum=momentum))

        

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')

        h2 = lrelu(tf.layers.batch_normalization(h2, training=is_training, momentum=momentum))

        

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')

        h3 = lrelu(tf.layers.batch_normalization(h3, training=is_training, momentum=momentum))



        h4 = tf.layers.flatten(h3)

        h4 = tf.layers.dense(h4, units=1)

        return tf.nn.sigmoid(h4), h4
def generator(z, is_training=is_training):

    momentum = 0.9

    with tf.variable_scope('generator', reuse=None):

        d = 4

        h0 = tf.layers.dense(z, units=d * d * 512)

        h0 = tf.reshape(h0, shape=[-1, d, d, 512])

        h0 = tf.nn.relu(tf.layers.batch_normalization(h0, training=is_training, momentum=momentum))

        

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')

        h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=is_training, momentum=momentum))

        

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')

        h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=is_training, momentum=momentum))

        

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')

        h3 = tf.nn.relu(tf.layers.batch_normalization(h3, training=is_training, momentum=momentum))

        

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh, name='g')

        return h4
g = generator(noise)

d_real, d_real_logits = discriminator(X)

d_fake, d_fake_logits = discriminator(g, reuse=True)



vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]

vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]



loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real)))

loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake)))

loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake)))

loss_d = loss_d_real + loss_d_fake
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):

    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)

    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)
def read_image(image_name, height, width):

    image = get_crop_image(image_name)

    

    h = image.size[0]

    w = image.size[1]

    

    image = np.array(image.resize((height, width)))

    return image / 255.
def montage(images):    

    if isinstance(images, list):

        images = np.array(images)

    img_h = images.shape[1]

    img_w = images.shape[2]

    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    if len(images.shape) == 4 and images.shape[3] == 3:

        m = np.ones(

            (images.shape[1] * n_plots + n_plots + 1,

             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5

    elif len(images.shape) == 4 and images.shape[3] == 1:

        m = np.ones(

            (images.shape[1] * n_plots + n_plots + 1,

             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5

    elif len(images.shape) == 3:

        m = np.ones(

            (images.shape[1] * n_plots + n_plots + 1,

             images.shape[2] * n_plots + n_plots + 1)) * 0.5

    else:

        raise ValueError('Could not parse image shape of {}'.format(images.shape))

    for i in range(n_plots):

        for j in range(n_plots):

            this_filter = i * n_plots + j

            if this_filter < images.shape[0]:

                this_img = images[this_filter]

                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,

                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img

    return m
batch_size = 4

epochs = 500000
# Tensorflow is dead when running a little time.

sv = tf.train.Supervisor()

saver = sv.saver

with sv.managed_session() as sess:

    z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

    samples = []

    loss = {'d': [], 'g': []}



    offset = 0

    for i in tqdm_notebook(range(epochs)):

        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)



        offset = (offset + batch_size) % len(images)

        batch = np.array([read_image(img, HEIGHT, WIDTH) for img in all_images[offset: offset + batch_size]])

        batch = (batch - 0.5) * 2



        d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})

        loss['d'].append(d_ls)

        loss['g'].append(g_ls)



        sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})

        sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})

        sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})



        if i % 10000 == 0:

            print(i, d_ls, g_ls)

            gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})

            gen_imgs = (gen_imgs + 1) / 2

            imgs = [img[:, :, :] for img in gen_imgs]

            gen_imgs = montage(imgs)

            plt.axis('off')

            plt.imshow(gen_imgs)

            plt.show()



    plt.plot(loss['d'], label='Discriminator')

    plt.plot(loss['g'], label='Generator')

    plt.legend(loc='upper right')

    plt.show()

    

    graph = tf.get_default_graph()

    g = graph.get_tensor_by_name('generator/g/Tanh:0')

    noise = graph.get_tensor_by_name('noise:0')

    is_training = graph.get_tensor_by_name('is_training:0')



    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

    gen_imgs = sess.run(g, feed_dict={noise: n, is_training: False})

    gen_imgs = (gen_imgs + 1) / 2

    imgs = [img[:, :, :] for img in gen_imgs]

    gen_imgs = montage(imgs)

    gen_imgs = np.clip(gen_imgs, 0, 1)

    plt.figure(figsize=(8, 8))

    plt.axis('off')

    plt.imshow(gen_imgs)

    plt.show()

    

    # Store data

    n_batches = 10000 // batch_size

    last_batch_size = 10000 % batch_size



    for i in tqdm_notebook(range(n_batches)):

        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

        gen_imgs = sess.run(g, feed_dict={noise: n, is_training: False})

        gen_imgs = (gen_imgs + 1) / 2

        for j in range(batch_size):

            imsave(os.path.join(GEN_DIR, f'sample_{i}_{j}.png'), gen_imgs[j])

    

    for i in range(last_batch_size):

        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

        gen_imgs = sess.run(g, feed_dict={noise: n, is_training: False})

        gen_imgs = (gen_imgs + 1) / 2

        imsave(os.path.join(GEN_DIR, f'sample_{n_batches}_{i}.png'), gen_imgs[i])
%%time

shutil.make_archive('images', 'zip', GEN_DIR)
len(os.listdir(GEN_DIR))
!rm -rf generated_dogs/

!rm -rf samples_dogs/
!ls -l .