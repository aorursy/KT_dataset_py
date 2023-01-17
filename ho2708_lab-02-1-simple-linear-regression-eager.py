# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
import numpy as np



print(tf.__version__)
x_data = [1, 2, 3, 4, 5]

y_data = [1, 2, 3, 4, 5]
import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'o')

plt.ylim(0, 8)
v = [1., 2., 3., 4.]

tf.reduce_mean(v) #2.5
tf.square(3) #9
x_data = [1, 2, 3, 4, 5]

y_data = [1, 2, 3, 4, 5]



W = tf.Variable(2.0)

b = tf.Variable(0.5)



hypothesis = W + x_data + b
W.numpy(), b.numpy()
hypothesis.numpy()
plt.plot(x_data, hypothesis.numpy(), 'r-')

plt.plot(x_data, y_data, 'o')

plt.ylim(0, 8)

plt.show()
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
with tf.GradientTape() as tape:

    hypothesis = W * x_data + b

    cost = tf.reduce_mean(tf.square(hypothesis - y_data))



W_grad, b_grad = tape.gradient(cost, [W, b])

W_grad.numpy(), b_grad.numpy()
learning_rate = 0.01



W.assign_sub(learning_rate * W_grad)

b.assign_sub(learning_rate * b_grad)



W.numpy(), b.numpy()
plt.plot(x_data, hypothesis.numpy(), 'r-')

plt.plot(x_data, y_data, 'o')

plt.ylim(0, 8)
W = tf.Variable(2.9)

b = tf.Variable(0.5)



for i in range(100):

    with tf.GradientTape() as tape:

        hypothesis = W * x_data + b

        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)

    b.assign_sub(learning_rate * b_grad)

    if i % 10 == 0:

      print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))



plt.plot(x_data, y_data, 'o')

plt.plot(x_data, hypothesis.numpy(), 'r-')

plt.ylim(0, 8)
print(W * 5 + b)

print(W * 2.5 + b)
import tensorflow as tf

import numpy as np



# Data

x_data = [1, 2, 3, 4, 5]

y_data = [1, 2, 3, 4, 5]



# W, b initialize

W = tf.Variable(2.9)

b = tf.Variable(0.5)



# W, b update

for i in range(100):

    # Gradient descent

    with tf.GradientTape() as tape:

        hypothesis = W * x_data + b

        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)

    b.assign_sub(learning_rate * b_grad)

    if i % 10 == 0:

      print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))



print()



# predict

print(W * 5 + b)

print(W * 2.5 + b)