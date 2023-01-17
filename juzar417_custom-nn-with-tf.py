# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))

#import tensorflow as tf

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

print(tf.__version__)

# Any results you write to the current directory are saved as output.
PATH = "../input/mnist_train.csv"

df = pd.read_csv(PATH)
c = df.columns[1:]
n_classes = 10

n_hidden_1 = 128

n_input = 784

total_len = 60000

batch_size = 1

n_iters = 10



lr = 0.05



total_batches = int(total_len/batch_size)



weights = {

    'w1' : tf.Variable(tf.random.normal([n_input, n_hidden_1])),

    'out' : tf.Variable(tf.random.normal([n_hidden_1, n_classes]))

}



bias = {

    'b1' : tf.Variable(tf.random.normal([n_hidden_1])),

    'out' : tf.Variable(tf.random.normal([n_classes]))

}



X = tf.placeholder("float", [batch_size, n_input])

Y = tf.placeholder(tf.int32, [batch_size, 1])
training_set = (

    tf.data.Dataset.from_tensor_slices(

        (

        tf.cast(df[c].values, tf.float32),

        tf.cast(df['label'].values, tf.int32)

        )

    )

).repeat().batch(batch_size)
iter = training_set.make_one_shot_iterator()

batch_x,batch_y = iter.get_next()
def mlp(x):

    x = tf.reshape(x, [batch_size, n_input])

    layer1 = tf.math.sigmoid(tf.add(tf.matmul(x, weights['w1']), bias['b1']))

    out_layer = tf.math.sigmoid(tf.add(tf.matmul(layer1, weights['out']), bias['out']))

    layers =  { 'l1':layer1,'out':out_layer}

    return layers







def computeCost(out, true_values):

    true_values = tf.one_hot(true_values ,n_classes, dtype=tf.float32)

    cost =tf.reduce_mean(tf.square(tf.subtract(true_values, out)))

    '''              C = 1/2m * sum(sum( a - y)^2)           '''

    return cost



def minimizeloss(x, y, learning_rate=0.001):

    logit = mlp(x)

    for i in range(batch_size):

        true_val = tf.reshape(y[i],[1,1])

        true_val = tf.one_hot(true_val,n_classes, dtype=tf.float32, axis=0)

        true_val = tf.reshape(true_val, [n_classes,1])

        gradC_activation = tf.math.subtract(tf.reshape(logit['out'][i,:], [n_classes, 1]),true_val)

        sigmoid_grad1 =  (tf.reshape(logit['out'][i,:], [n_classes, 1]))*(tf.math.subtract(1.0, tf.reshape(logit['out'][i,:], [n_classes, 1])))

        delta1 =  gradC_activation * sigmoid_grad1

        dCw5 = np.dot (delta1, tf.transpose(tf.reshape(logit['l1'][i,:], [n_hidden_1, 1])))

        dCb5 = tf.reshape(delta1, [n_classes])

        

        sigmoid_grad3 = tf.reshape(logit['l1'][i,:], [n_hidden_1,1])*(tf.math.subtract(1.0, tf.reshape(logit['l1'][i,:], [n_hidden_1, 1])))

        delta3 = tf.matmul(weights['out'], delta1) * sigmoid_grad3

        dCw1 = np.dot(delta3, tf.reshape(x[i], [1, n_input]))

        dCb1 = tf.reshape(delta3, [n_hidden_1])

        

        weights['w1'] =weights['w1'] - lr * tf.transpose(dCw1)

        weights['out'] = weights['out'] - lr * tf.transpose(dCw5)

        bias['b1'] = bias['b1'] - lr * dCb1

        bias['out'] = bias['out'] - lr * dCb5

        

    return tf.convert_to_tensor(1)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    Cost = []

    for i in range(1):

        y_ = batch_y.eval().reshape(batch_size, 1)

        x_ = batch_x.eval().reshape(batch_size, n_input)

        a  = sess.run([minimizeloss(X,Y, lr)], feed_dict={X:x_,Y:y_})

        batch_x,batch_y = iter.get_next()

        #Cost.append(c)

        #print("The loss is {}:".format(c))

        print("k")

print("Optimization Finished!")