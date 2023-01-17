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
n_inputs = 3  #Lets assume the RNN runs over only two time steps, taking input vectors of size 3 at each time step

n_neurons = 5 #RNN composed of a layer of five recurrent neurons
X0 = tf.placeholder(tf.float32,[None,n_inputs])

X1 = tf.placeholder(tf.float32,[None,n_inputs])
Wx = tf.Variable(tf.random_normal(shape=[n_inputs,n_neurons],dtype=tf.float32))

Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1,n_neurons],dtype=tf.float32))
Y0 = tf.tanh(tf.matmul(X0,Wx)+b)

Y1=tf.tanh(tf.matmul(Y0,Wy)+tf.matmul(X1,Wx)+b)
init = tf.global_variables_initializer()
#Mini Batch        Instance 0, Instance 1, Instance 2 , Instance 3

X0_batch = np.array([[0,1,2],    [3,4,5],   [6,7,8],     [9,0,1]])    #t=0

X1_batch = np.array([[9,8,7],    [0,0,0],   [6,5,4],     [3,2,1]])    #t=1
with tf.Session() as sess:

    init.run()

    Y0_val , Y1_val = sess.run([Y0 , Y1], feed_dict = {X0 : X0_batch , X1: X1_batch})

    
print(Y0_val)          # Output at t = 0
print(Y1_val)         # Output at t = 1