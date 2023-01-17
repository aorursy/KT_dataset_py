# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.misc

import scipy.io

import tensorflow as tf

from functools import reduce



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import math

from argparse import ArgumentParser

from PIL import Image

import matplotlib.pyplot as plt

import pylab



# Any results you write to the current directory are saved as output.
model_path = "../input/imagenet-vgg-verydeep-19.mat"

content_path = "../input/8.jpeg"

style_path = "../input/style1.jpg"



CONTENT_LAYERS = ('relu4_2', 'relu5_2')

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

pooling = 'max'
# 一些工具函数

def imread(path):

    img = scipy.misc.imread(path).astype(np.float)

    if len(img.shape) == 2:

        # grayscale

        img = np.dstack((img,img,img))

    elif img.shape[2] == 4:

        # PNG with alpha channel

        img = img[:,:,:3]

    return img



def _tensor_size(tensor):

    from operator import mul

    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
# vgg16 网络结构



VGG19_LAYERS = (

    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',



    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',



    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',

    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',



    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',

    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',



    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',

    'relu5_3', 'conv5_4', 'relu5_4'

)



def load_net(data_path):

    data = scipy.io.loadmat(data_path)

    if not all(i in data for i in ('layers', 'classes', 'normalization')):

        raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")

    mean = data['normalization'][0][0][0]

    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = data['layers'][0]

    return weights, mean_pixel



def net_preloaded(weights, input_image, pooling):

    net = {}

    current = input_image

    for i, name in enumerate(VGG19_LAYERS):

        kind = name[:4]

        if kind == 'conv':

            kernels, bias = weights[i][0][0][0][0]

            # matconvnet: weights are [width, height, in_channels, out_channels]

            # tensorflow: weights are [height, width, in_channels, out_channels]

            kernels = np.transpose(kernels, (1, 0, 2, 3))

            bias = bias.reshape(-1)

            current = _conv_layer(current, kernels, bias)

        elif kind == 'relu':

            current = tf.nn.relu(current)

        elif kind == 'pool':

            current = _pool_layer(current, pooling)

        net[name] = current



    assert len(net) == len(VGG19_LAYERS)

    return net



def _conv_layer(input, weights, bias):

    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),

            padding='SAME')

    return tf.nn.bias_add(conv, bias)





def _pool_layer(input, pooling):

    if pooling == 'avg':

        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),

                padding='SAME')

    else:

        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),

                padding='SAME')



def preprocess(image, mean_pixel):

    return image - mean_pixel





def unprocess(image, mean_pixel):

    return image + mean_pixel
content_image = imread(content_path)

style_image = imread(style_path)



style_image = scipy.misc.imresize(style_image, content_image.shape[1] / style_image.shape[1])

content_image = scipy.misc.imresize(content_image, 1.0)

print("style image size:", style_image.shape)



plt.figure(num='astronaut2',figsize=(32,32))

plt.subplot(6,6,1)

plt.imshow(content_image[:,:,:])

plt.subplot(6,6,2)

plt.imshow(style_image[:,:,:])
shape = (1,) + content_image.shape

style_shape = (1,) + style_image.shape

content_features = {}

style_features = {}



# 载入模型参数

vgg_weights, vgg_mean_pixel = load_net(model_path)



# 设置各层输出的权重

layer_weight = 1.0

style_layer_weight_exp = 1.0

style_layers_weights = {}

for style_layer in STYLE_LAYERS:

    style_layers_weights[style_layer] = layer_weight

    layer_weight *= style_layer_weight_exp



# 将权重归一化

layer_weights_sum = 0

for style_layer in STYLE_LAYERS:

    layer_weights_sum += style_layers_weights[style_layer]

for style_layer in STYLE_LAYERS:

    style_layers_weights[style_layer] /= layer_weights_sum

    

# 提取原始图第4、5卷积块的输出

g = tf.Graph()

with g.as_default(), tf.Session() as sess:

    image = tf.placeholder('float', shape=shape)

    net = net_preloaded(vgg_weights, image, pooling)

    content_pre = np.array([preprocess(content_image, vgg_mean_pixel)])

    for layer in CONTENT_LAYERS:

        content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

        

# 提取风格图的5个卷积块的输出

g = tf.Graph()

with g.as_default(), tf.Session() as sess:

    image = tf.placeholder('float', shape=style_shape)

    net = net_preloaded(vgg_weights, image, pooling)

    style_pre = np.array([preprocess(style_image, vgg_mean_pixel)])

    for layer in STYLE_LAYERS:

        feature = net[layer].eval(feed_dict={image: style_pre})

        feature = np.reshape(feature, (-1, feature.shape[3]))

        gram = np.matmul(feature.T, feature) / feature.size

        style_features[layer] = gram
# 初始化目标图

with tf.Graph().as_default():

    initial = tf.random_normal(shape) * 0.256

    image = tf.Variable(initial)



    net = net_preloaded(vgg_weights, image, pooling)



    # 原图损失值在第4、5卷积块的权重比例 

    content_layers_weights = {}

    content_layers_weights['relu4_2'] = 1.0

    content_layers_weights['relu5_2'] = 0.0



    # 计算目标图与原图之间的损失值

    content_weight = 5.0

    style_weight = 500.0

    tv_weight = 1e2

    content_loss = 0

    content_losses = []

    for content_layer in CONTENT_LAYERS:

        content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(

                    net[content_layer] - content_features[content_layer]) /

                    content_features[content_layer].size))

    content_loss += reduce(tf.add, content_losses)



    # 计算目标图与风格图之间的损失值

    style_loss = 0

    style_losses = []

    for style_layer in STYLE_LAYERS:

        layer = net[style_layer]

        i = 0;

        _, height, width, number = map(lambda i: i.value, layer.get_shape())

        size = height * width * number

        feats = tf.reshape(layer, (-1, number))

        gram = tf.matmul(tf.transpose(feats), feats) / size

        style_gram = style_features[style_layer]

        style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)

    style_loss = style_weight * reduce(tf.add, style_losses)

    

    # 这里是个去噪算法，不用纠结

    tv_y_size = _tensor_size(image[:,1:,:,:])

    tv_x_size = _tensor_size(image[:,:,1:,:])

    tv_loss = tv_weight * 2 * (

                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /

                    tv_y_size) +

                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /

                    tv_x_size))

    

    # overall loss

    loss = content_loss + style_loss + tv_loss

         

    # optimizer setup

    learning_rate = 10.0

    beta1 = 0.9

    beta2 = 0.999

    epsilon = 1e-08

    train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

    

    def print_progress():

        print('  iterations: %g\n' % i)

        print('  content loss: %g\n' % content_loss.eval())

        print('    style loss: %g\n' % style_loss.eval())

        print('       tv loss: %g\n' % tv_loss.eval())

        print('    total loss: %g\n' % loss.eval())

        print(' cur best loss: %g\n' % best_loss)

        

        img_show = np.array(img_out).astype(np.int32)

        name="showimg%(idx)d"%{'idx':i}

        plt.figure(num=name,figsize=(24,24))

        plt.subplot(1,3,1)

        #pylab.show()

        plt.imshow(img_show[:,:,:])



    

    best_loss = float('inf')

    best = None

    print_iterations = 10

    iterations = 500

    

    # 执行梯度下降

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print('Optimization started...\n')



        for i in range(iterations):

            train_step.run()



            last_step = (i == iterations - 1)

            if (print_iterations and i % print_iterations == 0):

                this_loss = loss.eval()

                if this_loss < best_loss:

                    best_loss = this_loss

                    best = image.eval()



                cur_img = image.eval();

                img_out = unprocess(cur_img.reshape(shape[1:]), vgg_mean_pixel)

                print_progress()

                

            if(last_step):

                img_out = unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                print_progress()



                    


