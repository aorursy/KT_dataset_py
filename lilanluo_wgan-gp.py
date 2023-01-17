# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf

# tf.enable_eager_execution()  # 可以实现立即输出，不用再麻烦地给Session()

import numpy as np

import os

import matplotlib.pyplot as plt

from tensorflow.keras import layers

from tensorflow import keras

from PIL import Image

%matplotlib inline

from imageio import imread, imsave, mimsave

import cv2

import glob

import shutil

import xml.etree.ElementTree as ET # xml parser used during pre-processing stage

import time # time the execution of codeblocks

import xml.dom.minidom # for printing the annotation xml nicely

print(os.listdir("../input"))

tf.__version__







# Any results you write to the current directory are saved as output.
BATCH_SIZE = 100

z_dim = 1000

WIDTH = 64

HEIGHT = 64

LAMBDA = 10

DIS_ITERS = 3 # 5

DEPTH = 3

OUTPUT_DIR = '../samples_dog' 

Model_Path = '../checkpoint'

DIRout = '../output_images'

    

if os.path.exists(DIRout):

    shutil.rmtree(DIRout)

if not os.path.exists(DIRout):

    os.mkdir(DIRout)

X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, HEIGHT, WIDTH, 3], name='X')

noise = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, z_dim], name='noise')

is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x, leak=0.2):

    return tf.maximum(x, leak * x)
def crop_images(image_dir, annotation_dir):

    IMAGES = os.listdir(image_dir)

    breeds = os.listdir(annotation_dir)  #狗狗种类

    idxIn = 0; namesIn = []

    imagesIn = np.zeros((25000,64,64,3))



    # CROP WITH BOUNDING BOXES TO GET DOGS ONLY

    # iterate through each directory in annotation

    for breed in breeds:

        # iterate through each file in the directory

        for dog in os.listdir(os.path.join(annotation_dir,breed)):

            try: img = Image.open(image_dir + '/'+dog+'.jpg') 

            except: continue           

            # Element Tree library allows for parsing xml and getting specific tag values    

            tree = ET.parse(annotation_dir+'/'+breed+'/'+dog)

            # take a look at the print out of an xml previously to get what is going on

            root = tree.getroot() # <annotation>

            objects = root.findall('object') # <object>

            for o in objects:

                bndbox = o.find('bndbox') # <bndbox>

                xmin = int(bndbox.find('xmin').text) # <xmin>

                ymin = int(bndbox.find('ymin').text) # <ymin>

                xmax = int(bndbox.find('xmax').text) # <xmax>

                ymax = int(bndbox.find('ymax').text) # <ymax>

                w = np.min((xmax - xmin, ymax - ymin))

                img2 = img.crop((xmin, ymin, xmin+w, ymin+w))

                img2 = img2.resize((64,64), Image.ANTIALIAS)

                imagesIn[idxIn,:,:,:] = np.asarray(img2)

                namesIn.append(breed)

                idxIn += 1   

#     print(idxIn)

    

    return imagesIn, namesIn, idxIn
def preprocess_image(images,idxIn):

    images = images[:idxIn,:,:,:] / 255.

    images = (images[:idxIn,:,:,:] - 0.5)*2

    return images
image_dir = '../input/all-dogs/all-dogs'

annotation_dir = '../input/annotation/Annotation'

imagesIn,namesIn, idxIn = crop_images(image_dir, annotation_dir)

print(idxIn)
x = np.random.randint(0,idxIn,25)



for k in range(5):

    plt.figure(figsize=(15,3))

    for j in range(5):

        plt.subplot(1,5,j+1)

        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )

        plt.axis('off')

        plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)

        plt.imshow(img)

    plt.show()
imagesIn = preprocess_image(imagesIn, idxIn)

imagesIn = imagesIn[:idxIn,:,:,:]

imageNums = imagesIn.shape[0]

print(imageNums)
def gernerate_shuffle_index(epoches, image_nums):

    shuffle_indice = np.random.permutation(np.arange(image_nums))

    for i in range(epoches-1):

        permu = np.random.permutation(np.arange(image_nums))

        shuffle_indice = np.concatenate((shuffle_indice, permu),axis=0)

    return shuffle_indice
def get_batch_image(shuffle_indice, begin, end):

    batch_index = shuffle_indice[begin:end]

    batch_image = np.array([imagesIn[i,:,:,:] for i in batch_index])

    return batch_image
def discriminator(image, reuse=None, is_training=is_training):

    momentum = 0.9

    with tf.variable_scope('discriminator', reuse=reuse):

        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=32, strides=2, padding='same'))

        

        h1 = lrelu(tf.layers.conv2d(h0, kernel_size=5, filters=64, strides=2, padding='same'))

        

        h2 = lrelu(tf.layers.conv2d(h1, kernel_size=5, filters=128, strides=2, padding='same'))

        

        h3 = lrelu(tf.layers.conv2d(h2, kernel_size=5, filters=256, strides=2, padding='same'))

        

        h4 = tf.contrib.layers.flatten(h3)

        h4 = tf.layers.dense(h4, units=1)

        return h4
def generator(z, is_training=is_training):

    momentum = 0.9

    with tf.variable_scope('generator', reuse=None):

        d = 8

        h0 = tf.layers.dense(z, units=d * d * 128)

        h0 = tf.reshape(h0, shape=[-1, d, d, 128])

        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=64, strides=2, padding='same')

        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=32, strides=2, padding='same')

        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        

#         h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=32, strides=2, padding='same')

#         h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))

        

        h4 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh, name='g')

        return h4
g = generator(noise)

d_real = discriminator(X)

d_fake = discriminator(g, reuse=True)



loss_d_real = -tf.reduce_mean(d_real)

loss_d_fake = tf.reduce_mean(d_fake)

loss_g = -tf.reduce_mean(d_fake)

loss_d = loss_d_real + loss_d_fake



alpha = tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)

interpolates = alpha * X + (1 - alpha) * g

grad = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]

slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))

gp = tf.reduce_mean((slop - 1.) ** 2)

loss_d += LAMBDA * gp



vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]

vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):

    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)

    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)
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
EPOCHES = 300

Iterations = np.floor(EPOCHES * imageNums / BATCH_SIZE)

Shuffle_Indice = gernerate_shuffle_index(EPOCHES, imageNums)

Iterations = int(Iterations)

print(type(Iterations))

print(Iterations) 

print(Shuffle_Indice.shape)
from tqdm import tqdm
sess = tf.Session()

sess.run(tf.global_variables_initializer())

z_samples = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, z_dim]).astype(np.float32)

samples = []

loss = {'d': [], 'g': []} 

for i in tqdm(range(60000)):

    for j in range(DIS_ITERS):

        n = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, z_dim]).astype(np.float32)

        batch_begin = i*BATCH_SIZE

        batch_end = (i+1)*BATCH_SIZE

        batch_images = get_batch_image(Shuffle_Indice,batch_begin, batch_end)            

        _, d_ls = sess.run([optimizer_d, loss_d], feed_dict={X: batch_images, noise: n, is_training: True})



    _, g_ls = sess.run([optimizer_g, loss_g], feed_dict={X: batch_images, noise: n, is_training: True})



    loss['d'].append(d_ls)

    loss['g'].append(g_ls)



    if i % 500 == 0:

        print(i, d_ls, g_ls)

        # 这边用来输出的图像不是随机产生的，是固定的，所以会发现生成的图都差不多hh

        gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})

        gen_imgs = (gen_imgs + 1) / 2

        imgs = [img[:, :, :] for img in gen_imgs]

        gen_imgs = montage(imgs)

        plt.axis('off')

        plt.imshow(gen_imgs)

        plt.show()

        samples.append(gen_imgs)

plt.plot(loss['d'], label='Discriminator')

plt.plot(loss['g'], label='Generator')

plt.legend(loc='upper right')

plt.show()
n = 10000

batch = 100

for i in tqdm(range(0, n, batch)):

    z = np.random.normal(0,1,size=(batch, z_dim))

    gen_imgs = sess.run(g, feed_dict={noise: z, is_training: False})

    gen_imgs = (gen_imgs + 1) / 2

    imgs = [img[:, :, :] for img in gen_imgs]

    for index, img in enumerate (imgs):

        img = (img *255.).astype('uint8')

        imsave(os.path.join(DIRout, 'image_' + str(i+index+1).zfill(5) + '.png'), img)

        if (i + index +1) == n:

            break

print(len(os.listdir(DIRout)))
if os.path.exists('images.zip'):

    os.remove('images.zip')

shutil.make_archive('images', 'zip', DIRout)

# shutil.rmtree(DIRout)

print("<END>")