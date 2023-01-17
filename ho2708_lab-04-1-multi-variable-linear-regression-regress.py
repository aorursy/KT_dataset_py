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
tf.random.set_seed(0)  # for reproducibility
x1_data = [1, 0, 3, 0, 5]

x2_data = [0, 2, 0, 4, 0]

y_data  = [1, 2, 3, 4, 5]



W1 = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))

W2 = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))

b  = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))



learning_rate = tf.Variable(0.001)



for i in range(1000+1):

    with tf.GradientTape() as tape:

        hypothesis = W1 * x1_data + W2 * x2_data + b

        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    W1_grad, W2_grad, b_grad = tape.gradient(cost, [W1, W2, b])

    W1.assign_sub(learning_rate * W1_grad)

    W2.assign_sub(learning_rate * W2_grad)

    b.assign_sub(learning_rate * b_grad)



    if i % 50 == 0:

        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(

          i, cost.numpy(), W1.numpy()[0], W2.numpy()[0], b.numpy()[0]))