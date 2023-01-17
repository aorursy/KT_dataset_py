from __future__ import print_function

import numpy as np

import cv2

import pandas as pd

import pydicom

import matplotlib.pyplot as plt

from PIL import Image

from matplotlib.patches import Polygon

from sklearn.model_selection import train_test_split



import os,time,cv2, sys, math

import tensorflow as tf

import tensorflow.contrib.slim as slim

import numpy as np

import time, datetime

import argparse

import random

import os, sys

import subprocess
def data_loader_csv(path,columns=None):

    data = pd.read_csv(path)

    if not columns==None:

        data = data.filter(columns)

    return data

    





data = data_loader_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

data.head(10)
def image_reader(image_path, show='False'):

    image_arr = pydicom.read_file(image_path)

    image_arr = image_arr.pixel_array

    if show:

        plt.imshow(image_arr,cmap='gray')

        plt.show()

    return image_arr


arr = image_reader('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm', show=True)
data_rsna = data_loader_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv', ['patientId','Target'])

data_rsna.head(10)



def one_hot_encoder(class_data, n_labels):

    label = np.array(class_data).reshape(-1)

    return np.eye(n_labels)[label]





def next_batch_generator(path,data, batch_size, resize_size, n_labels):

    ix = np.random.choice(np.arange(len(data)), batch_size)

    imgs =[]

    labels =[]

    for i in ix:

        array_img = image_reader(path+data[i][0]+'.dcm', show=False)

        img = Image.fromarray(array_img)

        img = img.resize(resize_size)

        array_img =np.array(img) / 255

        imgs.append(array_img)

        label = one_hot_encoder(data[i][1],n_labels)

        labels.append(label)

    imgs = np.array(imgs)

    imgs = imgs.reshape((batch_size,imgs.shape[1],imgs.shape[2],1))

    labels = np.array(labels)

    labels = labels.reshape((batch_size,n_labels))



    return imgs, labels
data = data_rsna.values

train_data, val_data = train_test_split(data, test_size=0.1)
train_x, train_y = next_batch_generator('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/',train_data,128,(224,224), 2)

val_x, val_y = next_batch_generator('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/',val_data,64,(224,224), 2)

def subsample(inputs, factor, scope=None):

    if factor == 1:

        return inputs

    else:

        return slim.max_pool2d(inputs, [1, 1], stride=factor)





    

def global_pooll(input_tensor, pool_op=tf.nn.avg_pool):

    shape = input_tensor.get_shape().as_list()

    if shape[1] is None or shape[2] is None:

        kernel_size = tf.convert_to_tensor([1, tf.shape(input_tensor)[1],tf.shape(input_tensor)[2], 1])

    else:

        kernel_size = [1, shape[1], shape[2], 1]

    output = pool_op(

      input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')

    # Recover output shape, for unknown shape.

    output.set_shape([None, 1, 1, None])

    return output

    

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):



    if stride == 1:

        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,

                           padding='SAME')

    else:

        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)

        pad_total = kernel_size_effective - 1

        pad_beg = pad_total // 2

        pad_end = pad_total - pad_beg

        inputs = tf.pad(inputs,

                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,

                           rate=rate, padding='VALID', weights_regularizer=slim.l2_regularizer(0.0001),weights_initializer=slim.variance_scaling_initializer())





def resBlock(x, Depth,depth_bottleneck, kernel_size=[3, 3], stride=1, skipCon=False, rate=2):

    depth_in = slim.utils.last_dimension(x.get_shape(), min_rank=4)

    peatct = slim.batch_norm(x, activation_fn=tf.nn.relu)

    if Depth == depth_in:

        shortcut = subsample(x, stride)

    else:

        shortcut = slim.conv2d(peatct, Depth, [1, 1], stride=stride,

                               activation_fn=None)

    residual = slim.conv2d(peatct, depth_bottleneck, [1, 1], stride=1, weights_regularizer=slim.l2_regularizer(0.0001),weights_initializer=slim.variance_scaling_initializer())

    residual = tf.nn.relu(slim.batch_norm(residual, fused=True, scale=True))

    residual = conv2d_same(residual, depth_bottleneck, 3, stride,rate=rate)

    residual = tf.nn.relu(slim.batch_norm(residual, fused=True, scale=True))

    residual = slim.conv2d(residual, Depth, [1, 1], stride=1,activation_fn=None)



    output = shortcut + residual

    return output





def UnitBlockA(x, base_depth, stride=1, rate=1):

    

    """

    A custom Block: Bekmirzaev shohrukh

    """

    

    depth =base_depth * 4

    depth_bottleneck =base_depth

    res = resBlock(x, depth, depth_bottleneck, stride=1, skipCon=True, rate=rate)

    res = resBlock(res, depth,depth_bottleneck, stride=1, skipCon=False, rate=rate)

    res = resBlock(res, depth,depth_bottleneck, stride=stride, skipCon=True, rate=rate)

    return res



def UnitBlockB(x, base_depth, stride=1, rate=1):

    

    """

    B custom Block: Bekmirzaev shohrukh

    

    """

    depth =base_depth * 4

    depth_bottleneck =base_depth

    res = resBlock(x, depth, depth_bottleneck, stride=1, skipCon=True, rate=rate)

    res = resBlock(res, depth,depth_bottleneck, stride=1, skipCon=False, rate=rate)

    res = resBlock(res, depth, depth_bottleneck, stride=1, skipCon=False, rate=rate)

    res = resBlock(res, depth,depth_bottleneck, stride=stride, skipCon=True, rate=rate)

    res = tf.nn.relu(res)

    return res









def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0, stride=1):

    out = slim.conv2d(inputs, n_filters, kernel_size, stride=stride, activation_fn=None, normalizer_fn=None)

    return out



def ourCustomNetwork(inputs,is_training=True,scope='OurCustomNetwork',num_classes=2):



    net = conv_block(inputs, 32, stride=2)

    net = UnitBlockA(net, 64, stride=2)

    net = UnitBlockB(net, 64, stride=2)

    net = UnitBlockA(net, 128, stride=2)

    net = UnitBlockB(net, 256, stride=2)

        

    with tf.variable_scope(scope):

        net = global_pooll(net)

        # 1 x 1 x num_classes

        # Note: legacy scope name.

        logits = slim.conv2d(

            net,

            num_classes, [1, 1],

            activation_fn=None,

            normalizer_fn=None,

            biases_initializer=tf.zeros_initializer(),

            scope='Conv2d_1c_1x1')

        logits = tf.squeeze(logits, [1, 2])

        logits = tf.identity(logits, name='output')

    return logits




net_input = tf.placeholder(tf.float32,shape=[None,224,224,1])

net_output = tf.placeholder(tf.float32,shape=[None,2])



def LOG(X, f=None):

    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    if not f:

        print(time_stamp + " " + X)

    else:

        f.write(time_stamp + " " + X)



def count_params():

    total_parameters = 0

    for variable in tf.trainable_variables():

        shape = variable.get_shape()

        variable_parameters = 1

        for dim in shape:

            variable_parameters *= dim.value

        total_parameters += variable_parameters

    print("This model has %d trainable parameters"% (total_parameters))
logits = ourCustomNetwork(net_input,scope='OurCustomNetwork', num_classes=2)
prediction = tf.nn.softmax(logits)



loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=net_output))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

train_op = optimizer.minimize(loss_op)



correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(net_output, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



init = tf.global_variables_initializer()
epochs = 5

batch_size =64

display_step = 100
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

avg_loss_per_epoch  = []

with tf.Session() as sess:

    saver=tf.train.Saver(max_to_keep=1000)

    sess.run(init)

    

    for epoch in range(epochs):

        num_steps = int(len(data) / batch_size)

        current_losses = []

        cnt=0

        st = time.time()

        epoch_st=time.time()



        for step in range(1, num_steps+1):

            train_x, train_y = next_batch_generator('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/',train_data,batch_size,(224,224), 2)

            _, loss, acc = sess.run([train_op,loss_op, accuracy], feed_dict={net_input: train_x, net_output:train_y})

            current_losses.append(loss)

            cnt = cnt + batch_size

            if cnt % 20 == 0:

                string_print = "Epoch = %d Count = %d Current_Loss = %.4f Current_Accuracy = %.2f Time = %.1f"%(epoch,cnt,loss,acc,time.time()-st)

                LOG(string_print)

                st = time.time()

                

        mean_loss = np.mean(current_losses)

        avg_loss_per_epoch.append(mean_loss)

        print("Saving latest checkpoint")

        model_checkpoint_name = "latest_model_" + str(epoch) + ".ckpt"

        #saver.save(sess,model_checkpoint_name)

        

        print("Validation Accuracy:")

        val_x, val_y = next_batch_generator('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/',val_data,64,(224,224), 2)

        accVal = sess.run(accuracy, feed_dict={net_input: val_x, net_output:val_y}) 

        print("Epoch " + str(epoch) +  ", Val Accuracy= " + "{:.3f}".format(accVal))

        

#         fig2, ax2 = plt.subplots(figsize=(11, 8))



#         ax2.plot(range(epoch+1), avg_loss_per_epoch)

#         ax2.set_title("Average loss vs epochs")

#         ax2.set_xlabel("Epoch")

#         ax2.set_ylabel("Current loss")



#         #plt.savefig('loss_vs_epochs.png')



#         plt.clf()

        

            