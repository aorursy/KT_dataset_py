import tensorflow as tf

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import glob

import seaborn as sns

import matplotlib.pyplot as plt

glob.glob('../input/*')
inputSize = 1

timeStep = 2
data = pd.read_csv('../input/airline-passengers.csv')['Passengers']/100

x = np.array([data[itr : itr+timeStep] for itr in range(len(data) - timeStep)]).reshape((-1, timeStep, inputSize))

y = np.array([data[itr+timeStep] for itr in range(len(data) - timeStep)]).reshape((-1, 1))
def model(x):

    

    cell = tf.nn.rnn_cell.BasicRNNCell(32)

    o, s = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)

    ls = tf.transpose(o, [1, 0, 2])[-1]

    dense = tf.layers.dense(ls, 1 )

    

    return dense
tf.reset_default_graph()

inRnn = tf.placeholder(tf.float32, [None ,timeStep, inputSize])

outRnn = tf.placeholder(tf.float32, [None, 1])



predModel = model(inRnn)

cost = tf.losses.mean_squared_error(predModel, outRnn)

opt = tf.train.RMSPropOptimizer(0.0001).minimize(cost)
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
for e in range(100):

    for i in range(len(x)):

        sess.run( opt, feed_dict={ inRnn : [x[i]],

                                   outRnn : [y[i]] } )

        

    _cost = sess.run(cost, feed_dict = { inRnn : [x[i]],

                                         outRnn : [y[i]] } )

    if(e%10==0):

        print(_cost)

        pred = sess.run(predModel, feed_dict = { inRnn : x })

        sns.lineplot(range(len(pred)), list(np.ravel(pred)))

        sns.lineplot(range(len(y)), list(np.ravel(y)), color='r')

        plt.show()