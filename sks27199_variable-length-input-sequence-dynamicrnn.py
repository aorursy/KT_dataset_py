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
tf.reset_default_graph()
n_steps = 2

n_inputs = 3

n_neurons = 5
seq_length = tf.placeholder(tf.int32,[None])
X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs , states = tf.nn.dynamic_rnn(basic_cell , X , dtype = tf.float32 , sequence_length=seq_length)
X_batch = np.array([

        # step 0     step 1

        [[0, 1, 2], [9, 8, 7]], # instance 0

        [[3, 4, 5], [0, 0, 0]], # instance 1 (padded with a zero vector)

        [[6, 7, 8], [6, 5, 4]], # instance 2

        [[9, 0, 1], [3, 2, 1]], # instance 3

    ])

seq_length_batch = np.array([2, 1, 2, 2])
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    outputs_val , states_val = sess.run([outputs,states] , feed_dict={X:X_batch , seq_length:seq_length_batch})
print(outputs_val)
print(states_val)