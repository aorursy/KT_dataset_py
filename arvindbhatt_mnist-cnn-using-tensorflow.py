# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
mnist=input_data.read_data_sets('/tmp/data',one_hot=True)
print(mnist)
print(mnist.train.images)
def weight_variable(shape):
    intial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(intial)
def bias_variable(shape):
    intial=tf.constant(0.1,shape=shape)
    return tf.Variable(intial)
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
    
def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def conv_layer(input,shape):
    x=shape[3];
    print(type(x))
    w=weight_variable(shape)
    b=bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input,w)+b)

def full_layer(input,size):
    in_size=int(input.get_shape()[1])
    w=weight_variable([in_size,size])
    b=bias_variable([size])
    return tf.matmul(input,w)+b
x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])
keep_prob=tf.placeholder(tf.float32,shape=[])
saver=tf.train.Saver()

x_image=tf.reshape(x,[-1,28,28,1])
conv1=conv_layer(x_image,[5,5,1,32])
conv1=max_pool_2(conv1)

conv1=conv_layer(conv1,[5,5,32,64])
conv1=max_pool_2(conv1)
conv1=tf.reshape(conv1,[-1,7*7*64])
conv1=full_layer(conv1,10)
y_conv=conv1



cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_p=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_p,tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch=mnist.train.next_batch(50)
        if i%10==0:
            print(sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5}))
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    
    saver.save(sess,"mymodel")
    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean([sess.run(accuracy,
    feed_dict={x:X[i], y_:Y[i],keep_prob:1.0})
                    for i in range(10)])
        #print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))
    print(test_accuracy)

mnist.train.next_batch(50)
