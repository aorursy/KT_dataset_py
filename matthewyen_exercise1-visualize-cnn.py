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
from keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()



"""show the shape of data"""

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

"""show first 10 images from the training dataset"""

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(x_train[i])
import tensorflow as tf



tf.reset_default_graph() # clean the graph we built before



NUM_CLASSES = 10



X = tf.placeholder(tf.float32, [None, 28, 28])

Y = tf.placeholder(tf.int64, [None])



X_extend = tf.reshape(X, [-1, 28, 28, 1])

Y_onehot = tf.one_hot(indices=Y, depth=NUM_CLASSES)



"""first convolution layer"""

conv1_w = tf.get_variable("conv1_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv1_b = tf.get_variable("conv1_b", [64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv1 = tf.nn.conv2d(X_extend, conv1_w, strides=[1,1,1,1], padding='SAME') + conv1_b

relu1 = tf.nn.relu(conv1)

pool1 = tf.nn.max_pool(value=relu1,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') # [None, 14, 14, 64]



"""second convolution layer"""

conv2_w = tf.get_variable("conv2_w", [3,3,64,64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv2_b = tf.get_variable("conv2_b", [64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1,1,1,1], padding='SAME') + conv2_b

relu2 = tf.nn.relu(conv2)

pool2 = tf.nn.max_pool(value=relu2,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') # [None, 7, 7, 64]



"""third convolution layer"""

conv3_w = tf.get_variable("conv3_w", [3,3,64,64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv3_b = tf.get_variable("conv3_b", [64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv3 = tf.nn.conv2d(pool2, conv3_w, strides=[1,1,1,1], padding='SAME') + conv3_b

relu3 = tf.nn.relu(conv3)



print(relu3)
"""flatten layer"""

flatten = tf.reshape(relu3, [-1, 7*7*64])



"""first fully connected layer"""

fc1 = tf.layers.dense(inputs=flatten,units=512,activation=tf.nn.relu,use_bias=True)



"""second fully connected layer"""

fc2 = tf.layers.dense(inputs=fc1,units=512,activation=tf.nn.relu,use_bias=True)



"""output layer"""

output = tf.layers.dense(inputs=fc2,units=NUM_CLASSES,activation=None,use_bias=True)





"""loss function"""

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot,logits=output))



"""accuracy function"""

accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), Y), dtype=tf.float32))



"""optimizer"""

opt = tf.train.AdamOptimizer(0.001).minimize(loss) # Global optimizer

"""Initiate the parameters"""

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
from tqdm import tqdm_notebook as tqdm



"""Training loop"""

EPOCHS = 10

BATCH_SIZE = 64



for epoch in range(0,EPOCHS):

    for step in tqdm(range(int(len(x_train)/BATCH_SIZE)), desc=('Epoch '+str(epoch))):

        """get next batch of training data"""

        x_batch = x_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        y_batch = y_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        """train"""

        loss_value, _ = sess.run([loss, opt], feed_dict={X: x_batch, Y: y_batch})

        

    """here, to make the code simple, I only use the first 1,000 images from testing dataset to test the network."""

    loss_value,accuracy_value = sess.run([loss, accu], feed_dict={X: x_test[:1000], Y: y_test[:1000]})

    print('Epoch loss: ', loss_value, '   accuracy: ', accuracy_value)

import numpy as np



"""extract first convolution layer filters"""

conv1_w_extract = sess.run(conv1_w)

print(conv1_w_extract.shape)



plt.figure(figsize=(20,20))

"""show first 10 filters"""

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(np.reshape(conv1_w_extract[:,:,:,i], [3,3]))
"""extract first convolution layer feature maps"""

conv1_fmaps = sess.run(relu1, feed_dict={X: [x_train[0]]})

print(conv1_fmaps.shape)



plt.figure(figsize=(20,20))

"""show first 10 fmaps"""

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(np.reshape(conv1_fmaps[0,:,:,i], [28,28]))