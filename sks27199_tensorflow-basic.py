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
import tensorflow as tf
x1=tf.constant(4.0,tf.float32)
x2=tf.constant(7.0)
print(x1,x2)

sess=tf.Session()
print(sess.run(x1))
print(sess.run([x1,x2]))
x3=tf.add(x1,x2)
print(sess.run(x3))
x4=tf.multiply(x1,x2)
print(sess.run(x4))
x4=tf.divide(x2,x1)
print(sess.run(x4))
x5=tf.subtract(x2,x1)
print(sess.run(x5))
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
addition_node=a+b
print(sess.run(addition_node,{a:7.0,b:11.1}))
print(sess.run(addition_node,{a:[7.0,32.4],b:[2,6.7]}))
add_and_double=addition_node*2
print(sess.run(add_and_double,{a:3.2,b:2}))
w=tf.Variable([3.0],tf.float32)
b=tf.Variable([2.0],tf.float32)
x=tf.placeholder(tf.float32)

model=w * x + b

init=tf.global_variables_initializer()
sess.run(init)

print(w)


print(sess.run(w))


print(sess.run(model,{x:[1,2,3]}))
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# print(sess.run(W)) would give an error because the variables are not initialized when you call tf.Variable

init = tf.global_variables_initializer()
sess.run(init)

print(W)
print(sess.run(W))

print(sess.run(linear_model, {x: [1,2,3,4]}))
m1 = tf.constant([[2,2]])
m2 = tf.constant([[3],[3]])
dot_operation = tf.matmul(m1,m2)

print(sess.run(dot_operation))
x1=tf.placeholder(dtype=float,shape=None)
y1=tf.placeholder(dtype=tf.float32,shape=None)
z1=x1+y1
x2=tf.placeholder(tf.float32,shape=[2,1])
y2=tf.placeholder(tf.float32,shape=[1,2])
z2=tf.matmul(x2,y2)
with tf.Session() as sess:
    #running one operation
    z1_value = sess.run(z1, feed_dict = { x1 : 5 , y1 : 6})
    print(z1_value)
    
    #running two operations at once
    z1_value , z2_value = sess.run( [ z1 , z2 ], feed_dict=
                                  {
                                      x1:4, y1:6,
                                      x2:[[2],[5]] , y2:[[3,4]]
                                  })
    print(z1_value)
    print(z2_value)
    
    




