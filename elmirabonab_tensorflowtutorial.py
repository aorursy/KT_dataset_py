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
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# define a computation graph and get shape and rank of a tensor

g = tf.Graph()
with g.as_default():
    t1 = tf.constant(np.pi) # scaler rank 0
    t2 = tf.constant([1,2])# vector rank 1
    t3 = tf.constant([[1,2],[4, np.nan]]) #matric rank 2
    t4 = tf.constant([[[1,2],[2,3]],[[4,5],[5,6]]])
    
    # get ranks
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)
    r4 = tf.rank(t4)
    # get their shape
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    s4 = t4.get_shape()
    
with tf.Session(graph=g) as sess:
    print('ranks without eval just returning tensor', r1)
    print('Ranks:', r1.eval(), r2.eval(), r3.eval(), r4.eval())
    print('Shape:', s1, s2, s3, s4)
    print('eval(t1)', t1.eval())
g = tf.Graph()
with g.as_default():
    a = tf.constant(1)
    b = tf.constant(10)
    tf_c = tf.placeholder(tf.int32, shape=[], name='tf_c')
    tf_d = tf.placeholder(tf.int32, shape=[], name= 'tf_d')
    h = a*tf_c + tf_d**b
with tf.Session(graph=g) as sess:
    feed = {tf_c:10, tf_d:5}
    print(a.eval())
    print('h', sess.run(h,feed_dict=feed) )
    
g = tf.Graph()
with g.as_default():
    a = tf.constant(1,name='a')
    b = tf.constant(2, name = 'b')
    c = tf. constant(3, name = 'c')
    
    r1 = a - b
    r2 = 2 * r1
    z = r2 + c
    # z = 2*(a-b)+c
with tf.Session(graph=g) as sess:
    print('2*(a-b)+c', z, z.eval())
# working with placeholders , variables and operations
# feeding data to model
g = tf.Graph()
with g.as_default():
    tf_a = tf.placeholder(tf.int32, shape=[], name='tf_a')
    tf_b = tf.placeholder(tf.int32, shape=[], name='tf_b')
    tf_c = tf.placeholder(tf.int32, shape=[], name='tf_c')
    
    r1 = tf_a - tf_b
    r2 = 2 * r1
    z = r2 + tf_c
with tf.Session(graph=g) as sess:
    feed = {tf_a:1,tf_b:2, tf_c:3}
    print('z:', sess.run(z, feed_dict=feed))

with tf.Session(graph=g) as sess:
    feed = {tf_a:1, tf_b:2}
    print(sess.run(r1, feed_dict=feed))
    print(sess.run(r2, feed_dict=feed))
g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(tf.float32, shape=[None, 2], name='tf_x')
    x_mean = tf.reduce_mean(tf_x, axis=0, name='mean')
np.random.seed(123)
np.set_printoptions(precision=2)

with tf.Session(graph=g) as sess:
    x1 = np.random.uniform(low=0, high=1, size=(5,2))
    print(x1.shape, x1)
    print(sess.run(x_mean, feed_dict={tf_x:x1}))
    
    x2= np.random.uniform(low=1, high=2, size=(10,2))
    print(x2.shape,x2)
    print(sess.run(x_mean, feed_dict={tf_x:x2}))
    print(tf_x)
    print(tf_x.get_shape(), tf_x.eval(feed_dict={tf_x:x1}))  #tensorflow can be evaluated GIVEN feed_dict
#variables
g1 = tf.Graph()

with g1.as_default():
    w = tf.Variable(np.array([[1,2,3],[4,5,6]]), name='w'
                   )
    z= tf.Variable(dtype=tf.int32, initial_value=np.array([[1,20,3.2],[4,5,6]]), name='z')

    print(w,z)
g1=tf.Graph()
with g1.as_default():
    w = tf.Variable(initial_value=[1,2], name='w')
with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))
g1 = tf.Graph()
with g1.as_default():
    w = tf.Variable([1,2], name='w')
    init_op = tf.global_variables_initializer()
    
with tf.Session(graph=g1) as sess:
    sess.run(init_op)
    print(sess.run(w))
g2 = tf.Graph()
with g2.as_default():
    w1 = tf.Variable(1, name='w1')
    init_op = tf.global_variables_initializer()
    w2= tf.Variable(2, name='w2')
with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print(sess.run(w1))
#     print(sess.run(w2)) # error if it's not initialized
g = tf.Graph()
with g.as_default():
    with tf.variable_scope('net_A'):
        with tf.variable_scope('layer-1'):
            w1 = tf.Variable(tf.random_normal(shape=(10,4)), name='weights')
        with tf.variable_scope('layer-2'):
            w2= tf.Variable(tf.random_normal(shape=(20,10)), name='weights')
    with tf.variable_scope('net_B'):
        with tf.variable_scope('layer-1'):
            w3= tf.Varaible(tf.random_normal(shape=(10,4)), name='weights')

