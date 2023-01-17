# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_in = np.loadtxt('../input/in.csv', delimiter=',', dtype='float32')

data_out = np.loadtxt('../input/out.csv', delimiter=',', dtype='float32')



data_in = data_in / 255

data_out = data_out / 255



data_in = data_in.reshape(-1, 74, 54, 3)

data_out = data_out.reshape(-1, 74, 54, 3)



print(data_in.dtype)
'''

A CNN model for enhancing image.

Input: Batch of images of shape (74, 54, 3)

Output: Batch of images of shape (74, 54, 3)

'''



#Input image batch of shape (74, 54, 3) of each

X = tf.placeholder(tf.float32, [None, 74, 54, 3])



#Output image batch of shape (74, 54, 3) of each

Y = tf.placeholder(tf.float32, [None, 74, 54, 3])

YY = tf.reshape(Y, shape=[-1, 74*54*3])



K = 4   # first convolutional layers output channels

L = 8   # second convolutional layers output channels

M = 12  # third convolutional layers output channels

N = 10  # fourth convolutional layers output channels

O = 6   # fifth convolutional layers output channels

P = 3   # sixth convolutional layers output channels



# Wieghts and biases of each layers

# Layer 1

W1 = tf.Variable(tf.constant(0.1, shape=[5, 5, 3, K], dtype=tf.float32)) # 5x5 patch, 3 input channels, K output channels

B1 = tf.Variable(tf.ones([K])/10)



# Layer 2

W2 = tf.Variable(tf.constant(0.1, shape=[5, 5, K, L], dtype=tf.float32))

B2 = tf.Variable(tf.ones([L])/10)



# Layer 3

W3 = tf.Variable(tf.constant(0.1, shape=[5, 5, L, M], dtype=tf.float32))

B3 = tf.Variable(tf.ones([M])/10)



# Layer 4

W4 = tf.Variable(tf.constant(0.1, shape=[5, 5, M, N], dtype=tf.float32))

B4 = tf.Variable(tf.ones([N])/10)



# Layer 5

W5 = tf.Variable(tf.constant(0.1, shape=[5, 5, N, O], dtype=tf.float32))

B5 = tf.Variable(tf.ones([O])/10)



# Layer 6

W6 = tf.Variable(tf.constant(0.1, shape=[5, 5, O, P], dtype=tf.float32))

B6 = tf.Variable(tf.ones([P])/10)



# The Model

# Stride is always one

Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)

Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)

Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)

Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, 1, 1, 1], padding='SAME') + B4)

Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1, 1, 1, 1], padding='SAME') + B5)



Y6 = tf.nn.conv2d(Y5, W6, strides=[1, 1, 1, 1], padding='SAME') + B6

Ylogits = tf.reshape(Y6, shape=[-1, 74*54*P])



MSE = tf.subtract(YY, Ylogits)

MSE = tf.square(MSE)

cross_entropy = tf.reduce_mean(MSE)*100



def train(eta, epochs, batch_size, train_in, train_out, test_data):

    optimizer = tf.train.AdamOptimizer(eta)

    train_step = optimizer.minimize(cross_entropy)



    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())



        for i in range(epochs):

            m_batch_in = [train_in[x: batch_size] for x in range(0, len(train_in), batch_size)]

            m_batch_out = [train_out[x: batch_size] for x in range(0, len(train_out), batch_size)]

            for train_x, train_y in zip(m_batch_in, m_batch_out):

                sess.run(train_step, feed_dict={X: train_x, Y: train_y})

            print('Epoch', i, 'completed...')

        

        outputs = sess.run(Y6, feed_dict={X: test_data})

        return outputs





from scipy import misc



test_data = data_in[-10:]

test_out = data_out[-10:]



print('Started...')



pred = train(0.001, 1000, 100, data_in[:-10], data_out[:-10], test_data)



for i in range(len(pred)):

    misc.imsave('out' + str(i) + '.jpg', pred[i])

    misc.imsave('in' + str(i) + '.jpg', test_out[i])

print('Completed...\nThanks Kaggle')