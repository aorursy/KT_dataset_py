import numpy as np

import pandas as pd

from IPython.display import display, HTML

# read dataset from download file

dataname = ['id','diagnosis','radius_mean','texture_mean','perimeter_mean',

            'area_mean','smoothness_mean','compactness_mean','concavity_mean',

            'concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se',

            'texture_se','perimeter_se','area_se','smoothness_se','compactness_se',

            'concavity_se','concave points_se','symmetry_se','fractal_dimension_se',

            'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',

            'compactness_worst','concavity_worst','concave points_worst','symmetry_worst',

            'fractal_dimension_worst']



data = pd.read_csv("../input/data.csv", header = 0)

data.shape

#display(data)

#HTML(data.to_html())
from IPython.display import display

data = data.drop("Unnamed: 32", axis =1 )

display(data.head())
nddata = data

import tensorflow as tf

def changetypeofcolumn(nddata):

    for i in range(len(nddata)):

        if nddata[dataname[1]][i] == "M":

            nddata[dataname[1]][i] = 1;

        elif nddata[dataname[1]][i] == "B":

            nddata[dataname[1]][i]= 0;

    databefore = pd.concat([nddata[dataname[0]], nddata[dataname[1]]], axis = 1)

    nddata = pd.concat([databefore, nddata[dataname[2:32]]], axis = 1)

    return nddata



xnddata = changetypeofcolumn(nddata)

nddata = xnddata[0:469]

nddata.shape

#display(nddata)



trnddata = xnddata[469:569]

trnddata.shape

#display(trnddata)
import tensorflow as tf

sess = tf.InteractiveSession()

graph = tf.Graph()



x = tf.placeholder("float", [100, 30])

y_ = tf.placeholder("float", [100, 1])

W = tf.Variable(tf.zeros([30, 1]))

b = tf.Variable(tf.zeros([1,1]))



y = tf.nn.softmax(tf.matmul(x, W) + b)



loss = tf.reduce_mean(tf.square(abs(y - y_)))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)



writer = tf.train.SummaryWriter('./P5tuning', sess.graph)
sess.run(tf.initialize_all_variables())


for i in range(1000):

    tfdata = nddata.sample(frac=1)

    ntfdata = tfdata[dataname[1:32]]

    traindata = ntfdata[:100]

    x_train = np.float32(traindata[dataname[2:32]])

    y_train = np.float32(traindata[dataname[1]])

    x_test = np.reshape(x_train, (100, 30))

    y_test = np.reshape(y_train, (100, 1))

    sess.run(optimizer, feed_dict={x:x_test, y_:y_test})

    
correct_production = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_production, "float"))

traindata = trnddata

x_train = np.float32(traindata[dataname[2:32]])

y_train = np.float32(traindata[dataname[1]])

x_test = np.reshape(x_train, (100, 30))

y_test = np.reshape(y_train, (100, 1))

results = sess.run(accuracy, feed_dict={x:x_test, y_:y_test})

print (results)