# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



tf.compat.v1.disable_eager_execution()

tf.compat.v1.set_random_seed(777)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
xy = pd.read_csv('/kaggle/input/heights-and-weights/data.csv')

xy.head()

x_data = xy.Height

y_data = xy.Weight
plt.title("Height and weight data")

plt.xlabel("Height")

plt.ylabel("Weight")

plt.scatter(x_data, y_data, color='red')
W = tf.Variable(tf.compat.v1.random_normal([1]), name = 'weight')

b = tf.Variable(tf.compat.v1.random_normal([1]), name = 'bias')

X = tf.compat.v1.placeholder(tf.float32, shape = [None])

Y = tf.compat.v1.placeholder(tf.float32, shape = [None])
hypothesis = X * W + b



cost = tf.reduce_mean(tf.square(hypothesis - Y))



optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)

train = optimizer.minimize(cost)
sess = tf.compat.v1.Session()



sess.run(tf.compat.v1.global_variables_initializer())
for step in range(20001):

    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X: x_data, Y: y_data})

    

    if step % 1000 == 0:

        print(step, cost_val, W_val, b_val)

        plt.xlabel('Height')

        plt.ylabel('Weight')

        plt.scatter(x_data, y_data, color = 'red', label = 'Original data')

        plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label = 'Hypothesis')

        plt.legend()

        plt.show()
print('예상한 W의 값은 ', sess.run(W), '예상한 b의 값은 ', sess.run(b))

plt.xlabel('x_data')

plt.ylabel('y_data')

plt.plot(x_data, y_data, 'ro', label='Original data')

plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='hypothesis')

plt.show()



print('어떤 사람의 Height = 1.5이면, Weight = ', sess.run(W) * 1.5 + sess.run(b))