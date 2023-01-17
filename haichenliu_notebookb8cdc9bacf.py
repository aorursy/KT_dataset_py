# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df=pd.read_csv('../input/train.csv')

images = df.values[:, 1:]

labels = df.values[:, 0]

print(images.shape, labels.shape)



df=pd.read_csv('../input/test.csv')

images_test = df.values



print(images_test.shape)



N = 42000

L = 784

C = 10



N_test = 28000

# Any results you write to the current directory are saved as output.
import tensorflow as tf

import numpy as np

import math

import timeit

import matplotlib.pyplot as plt

%matplotlib inline
D = 1024

batch_size = 200



def Random_Batch():

    index = np.random.randint(0, 42000, batch_size)

    batch_images = images[index]

    batch_labels = labels[index]

    return batch_images, batch_labels
H = 128

w1 = tf.Variable(tf.random_normal((L, H)))

b1 = tf.Variable(tf.zeros(H))

w2 = tf.Variable(tf.random_normal((H, C)))

b2 = tf.Variable(tf.zeros(C))

x = tf.placeholder(tf.float32, shape = (None, L))

y = tf.placeholder(tf.int64, shape = (None))

a = tf.add(tf.matmul(x, w1), b1)

h = tf.nn.relu(a)

out = tf.add(tf.matmul(h, w2), b2)

y_pred = tf.argmax(out, 1)



mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))



optimizer = tf.train.RMSPropOptimizer(learning_rate = 1e-2)



step = optimizer.minimize(mean_loss)



correct_prediction = tf.equal(y, y_pred)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())



for i in range(500):

    batch_xs, batch_ys = Random_Batch()

    loss, _ = sess.run([mean_loss, step], feed_dict={x: batch_xs, y: batch_ys})

    if i%100 == 0:

        print("loss: ", loss)
accuracy, y_pred = sess.run([accuracy,y_pred], feed_dict={x: images[0:100], y: labels[0:100]})

print(y_pred)

print(accuracy)

for i in range(10):

    sub = plt.subplot(1, 10, i + 1)

    plt.imshow(images[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
predict = tf.argmax(out, 1)

predict = sess.run([predict], feed_dict={x: images_test})

pred_array = np.asarray(predict[0])



for i in range(10):

    sub = plt.subplot(1, 10, i + 1)

    plt.imshow(images_test[i].reshape(28, 28), cmap=plt.get_cmap('gray'))