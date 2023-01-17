# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train = np.array(train.values)

test = np.array(test.values)



print(train.shape)

print(test.shape)
m = train.shape[0]

train_x = train[:,1:]

train_y = train[:,0].reshape((m, 1))



print(train_x.shape)

print(train_y.shape)
# normalize values



train_x = train_x/255

test_x = test/255
import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline



rand_i = np.random.randint(m, size=6)



_, subplots = plt.subplots(3, 2)

for i, img_i in enumerate(rand_i):

    img = (1 - train_x[img_i]).reshape(28,28)

    plt.subplot(3, 2, i + 1)

    plt.imshow(img, cmap='gray')

    plt.axis("off")

    plt.title("Label: {0}".format(train_y[img_i][0]))

train_y_onehot = np.zeros((m, 10))





import tensorflow as tf



learning_rate = 0.05



W = tf.Variable(np.random.randn(784, 10) * 0.001, name="W")

b = tf.Variable(np.zeros((m, 10)), name="b")



X = tf.placeholder(tf.float64, shape=[m, 784])

Y_ = tf.placeholder(tf.float64, shape=[None, 10])



Y = tf.nn.softmax(tf.matmul(X, W) + b)



cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))



optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init)

    sess.run(optimizer, feed_dict={ X: train_x, Y: train_y_onehot })