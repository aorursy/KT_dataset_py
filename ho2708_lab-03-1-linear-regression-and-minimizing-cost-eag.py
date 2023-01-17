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
import numpy as np



X = np.array([1, 2, 3])

Y = np.array([1, 2, 3])



def cost_func(W, X, Y):

    c = 0

    for i in range(len(X)):

        c += (W + X[i] - Y[i]) ** 2

    return c / len(X)



for feed_W in np.linspace(-3, 5, num=15):

    curr_cost = cost_func(feed_W, X, Y)

    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
X = np.array([1, 2, 3])

Y = np.array([1, 2, 3])



def cost_fuuc(W, X, Y):

    hypothesis = X * W

    return tf.reduce_mean(tf.square(hypothesis - Y))



W_values = np.linspace(-3, 5, num=15)

cost_values = []



for feed_W in W_values:

    curr_cost = cost_func(feed_W, X, Y)

    cost_values.append(curr_cost)

    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8,6)
import matplotlib.pyplot as plt



plt.plot(W_values, cost_values, "b")

plt.ylabel('Cost(W)')

plt.xlabel('W')

plt.show()
tf.random.set_seed(0)  # for reproducibility
x_data = [1., 2., 3., 4.]

y_data = [1., 3., 5., 7.]



W = tf.Variable(tf.random.normal((1,), -100., 100.))



for step in range(300):

    hypothesis = W * X

    cost = tf.reduce_mean(tf.square(hypothesis - Y))



    alpha = 0.01

    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))

    descent = W - tf.multiply(alpha, gradient)

    W.assign(descent)

    

    if step % 10 == 0:

        print('{:5} | {:10.4f} | {:10.6f}'.format(

            step, cost.numpy(), W.numpy()[0]))
print(5.0 * W)

print(2.5 * W)
x_data = [1., 2., 3., 4.]

y_data = [1., 3., 5., 7.]



W = tf.Variable([5.0])



for step in range(300):

    hypothesis = W * X

    cost = tf.reduce_mean(tf.square(hypothesis - Y))



    alpha = 0.01

    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))

    descent = W - tf.multiply(alpha, gradient)

    W.assign(descent)

    

    if step % 10 == 0:

        print('{:5} | {:10.4f} | {:10.6f}'.format(

            step, cost.numpy(), W.numpy()[0]))