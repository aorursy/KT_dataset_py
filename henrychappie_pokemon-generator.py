import numpy as np 

import pandas as pd 



import os

#数据所在的目录

print(os.listdir("../input/kaggle-one-shot-pokemon/kaggle-one-shot-pokemon"))
#重置图片的大小

import os

import cv2

#图片数据所在的位置

src = "../input/kaggle-one-shot-pokemon/kaggle-one-shot-pokemon/pokemon-b"

#重置后图片保存的位置

dst = "./resizedData" # resized



for each in os.listdir(src):

    img = cv2.imread(os.path.join(src,each))

    img = cv2.resize(img,(256,256))

    cv2.imwrite(os.path.join(dst,each), img)

#读取处理后的数据集

from glob import glob

import cv2

PATH = os.path.abspath(os.path.join( os.getcwd(), 'resizedData'))

IMGS = glob(os.path.join(PATH, "*.jpg"))
#导包

import tensorflow as tf

import random

import scipy.misc

from utils import *

slim = tf.contrib.slim
#设置预训练图片的大小以及颜色通道数

HEIGHT, WIDTH, CHANNEL = 128, 128, 3

#批处理的大小

BATCH_SIZE = 64

#训练的回合数

EPOCH = 5000

version = 'newPokemon'

#生成图片的位置

newPoke_path = './' + version
#生成的新的神奇宝贝所在的目录

newPoke_path
#查看生成的新神奇宝贝

PATH = os.path.abspath(os.path.join( './', 'newPokemon'))

IMGS = glob(os.path.join(PATH, "*.jpg"))

for i in range(0,5000,50):

    try:

        imgs = plt.imread('/kaggle/working/newPokemon/epoch{}.jpg'.format(i))

        #plt.imshow(imgs)

        #plt.show()

    except:

        pass

#print(len(IMGS))
#激活函数

def lrelu(x, n, leak=0.2): 

    return tf.maximum(x, leak * x, name=n) 
#数据处理

def process_data():   

    current_dir = os.getcwd()

    # parent = os.path.dirname(current_dir)

    pokemon_dir = os.path.join(current_dir, 'resizedData')

    images = []

    for each in os.listdir(pokemon_dir):

        images.append(os.path.join(pokemon_dir,each))

    # print images    

    all_images = tf.convert_to_tensor(images, dtype = tf.string)

    

    images_queue = tf.train.slice_input_producer(

                                        [all_images])

                                        

    content = tf.read_file(images_queue[0])

    image = tf.image.decode_jpeg(content, channels = CHANNEL)

    # sess1 = tf.Session()

    # print sess1.run(image)

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta = 0.1)

    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 

    # print image.get_shape()

    size = [HEIGHT, WIDTH]

    image = tf.image.resize_images(image, size)

    image.set_shape([HEIGHT,WIDTH,CHANNEL])

    # image = image + noise

    # image = tf.transpose(image, perm=[2, 0, 1])

    # print image.get_shape()

    

    image = tf.cast(image, tf.float32)

    image = image / 255.0

    

    iamges_batch = tf.train.shuffle_batch(

                                    [image], batch_size = BATCH_SIZE,

                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,

                                    min_after_dequeue = 200)

    num_images = len(images)



    return iamges_batch, num_images
#生成器

def generator(input, random_dim, is_train, reuse=False):

    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num

    s4 = 4

    output_dim = CHANNEL  # RGB image

    with tf.variable_scope('gen') as scope:

        if reuse:

            scope.reuse_variables()

        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))

        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,initializer=tf.constant_initializer(0.0))

        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

         #Convolution, bias, activation, repeat! 

        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')

        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')

        act1 = tf.nn.relu(bn1, name='act1')

        # 8*8*256

        #Convolution, bias, activation, repeat! 

        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                           name='conv2')

        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')

        act2 = tf.nn.relu(bn2, name='act2')

        # 16*16*128

        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                           name='conv3')

        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')

        act3 = tf.nn.relu(bn3, name='act3')

        # 32*32*64

        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                           name='conv4')

        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')

        act4 = tf.nn.relu(bn4, name='act4')

        # 64*64*32

        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                           name='conv5')

        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')

        act5 = tf.nn.relu(bn5, name='act5')

        

        #128*128*3

        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                           name='conv6')

        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')

        act6 = tf.nn.tanh(conv6, name='act6')

        return act6

#辨识器

def discriminator(input, is_train, reuse=False):

    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512

    with tf.variable_scope('dis') as scope:

        if reuse:

            scope.reuse_variables()



        #Convolution, activation, bias, repeat! 

        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                 name='conv1')

        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')

        act1 = lrelu(conv1, n='act1')

         #Convolution, activation, bias, repeat! 

        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                 name='conv2')

        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')

        act2 = lrelu(bn2, n='act2')

        #Convolution, activation, bias, repeat! 

        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                 name='conv3')

        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')

        act3 = lrelu(bn3, n='act3')

         #Convolution, activation, bias, repeat! 

        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",

                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),

                                 name='conv4')

        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')

        act4 = lrelu(bn4, n='act4')

       

        # start from act4

        dim = int(np.prod(act4.get_shape()[1:]))

        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

      

        

        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,

                             initializer=tf.truncated_normal_initializer(stddev=0.02))

        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,

                             initializer=tf.constant_initializer(0.0))



        # wgan just get rid of the sigmoid

        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')

        # dcgan

        acted_out = tf.nn.sigmoid(logits)

        return logits #, acted_out

#训练函数

def train():

    random_dim = 100

    

    with tf.variable_scope('input'):

        #real and fake image placholders

        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')

        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')

        is_train = tf.placeholder(tf.bool, name='is_train')

    

    # wgan

    fake_image = generator(random_input, random_dim, is_train)

    

    real_result = discriminator(real_image, is_train)

    fake_result = discriminator(fake_image, is_train, reuse=True)

    

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.

    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.

            



    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'dis' in var.name]

    g_vars = [var for var in t_vars if 'gen' in var.name]

    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)

    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)

    # clip discriminator weights

    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]



    

    batch_size = BATCH_SIZE

    image_batch, samples_num = process_data()

    

    batch_num = int(samples_num / batch_size)

    total_batch = 0

    sess = tf.Session()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())

    # continue training

    save_path = saver.save(sess, "/tmp/model.ckpt")

    ckpt = tf.train.latest_checkpoint('./model/' + version)

    saver.restore(sess, save_path)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)



    print('total training sample num:%d' % samples_num)

    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))

    print('start training...')

    for i in range(EPOCH):

        print("Running epoch {}/{}...".format(i, EPOCH))

        for j in range(batch_num):

            d_iters = 5

            g_iters = 1



            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)

            for k in range(d_iters):

                train_image = sess.run(image_batch)

                #wgan clip weights

                sess.run(d_clip)

                

                # Update the discriminator

                _, dLoss = sess.run([trainer_d, d_loss],

                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})



            # Update the generator

            for k in range(g_iters):

                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)

                _, gLoss = sess.run([trainer_g, g_loss],

                                    feed_dict={random_input: train_noise, is_train: True})



            #print('train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss))

            

        # save check point every 500 epoch

        if i%500 == 0:

            if not os.path.exists('./model/' + version):

                os.makedirs('./model/' + version)

            saver.save(sess, './model/' +version + '/' + str(i))  

        if i%50 == 0:

            # save images

            if not os.path.exists(newPoke_path):

                os.makedirs(newPoke_path)

            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)

            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})

            # imgtest = imgtest * 255.0

            # imgtest.astype(np.uint8)

            save_images(imgtest, [8,8] ,newPoke_path + '/epoch' + str(i) + '.jpg')

            plt.imread(newPoke_path + '/epoch' + str(i) + '.jpg')

            plt.imshow(imgs)

            plt.title('Generate Image After Epoch/0 Trained')

            plt.axis('off')

            plt.show()

            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))

            print('-------------------------------------------------------------------------')

    coord.request_stop()

    coord.join(threads)
#图片处理函数

from __future__ import division

import math

import json

import random

import pprint

import scipy.misc

import numpy as np

from time import gmtime, strftime

from six.moves import xrange



import tensorflow as tf

import tensorflow.contrib.slim as slim



pp = pprint.PrettyPrinter()



get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])



def show_all_variables():

    model_vars = tf.trainable_variables()

    slim.model_analyzer.analyze_vars(model_vars, print_info=True)



def get_image(image_path, input_height, input_width,

              resize_height=64, resize_width=64,

              crop=True, grayscale=False):

    image = imread(image_path, grayscale)

    return transform(image, input_height, input_width,

                   resize_height, resize_width, crop)



def save_images(images, size, image_path):

    return imsave(inverse_transform(images), size, image_path)



def imread(path, grayscale = False):

    if (grayscale):

        return scipy.misc.imread(path, flatten = True).astype(np.float)

    else:

        return scipy.misc.imread(path).astype(np.float)



def merge_images(images, size):

    return inverse_transform(images)



def merge(images, size):

    h, w = images.shape[1], images.shape[2]

    if (images.shape[3] in (3,4)):

        c = images.shape[3]

        img = np.zeros((h * size[0], w * size[1], c))

        for idx, image in enumerate(images):

            i = idx % size[1]

            j = idx // size[1]

            img[j * h:j * h + h, i * w:i * w + w, :] = image

        return img

    elif images.shape[3]==1:

        img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):

            i = idx % size[1]

            j = idx // size[1]

            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]

        return img

    else:

        raise ValueError('in merge(images,size) images parameter '

                     'must have dimensions: HxW or HxWx3 or HxWx4')



def imsave(images, size, path):

    image = np.squeeze(merge(images, size))

    return scipy.misc.imsave(path, image)
import time
if __name__ == "__main__":

    start_time = time.time()

    tf.reset_default_graph()

    train()

    end_time = time.time()

    print('共用时',end_time-start_time,'s')