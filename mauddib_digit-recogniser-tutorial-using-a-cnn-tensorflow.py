# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plot library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#! /usr/bin/env python

# -*- coding: utf-8 -*- 

import tensorflow as tf

import pandas as pd

import numpy as np

import pickle

from keras.utils.np_utils import to_categorical   





train_data0 = pd.read_csv("../input/train.csv")

train_data = np.array(train_data0.iloc[:, 1:785])

train_label = np.array(train_data0.iloc[:,0])





# One-hot encoding

train_label = to_categorical(train_label, num_classes=10)



test_data0 = pd.read_csv("../input/test.csv")

test_data = np.array(test_data0)





print(np.array(train_data0.iloc[:,0])[:10])

print('is equivalent to:')

print(train_label[:10])


pixels = train_data[9].reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()



pixels = train_data[7].reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()



pixels = train_data[6].reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()



# Define hyperparameters

learning_rate = 0.0001

epoch = 7

batch_size = 20



# Define network parameters

n_input = 784

n_classes = 10



# Placeholder

X = tf.placeholder(tf.float32, [None, n_input])

y = tf.placeholder(tf.float32, [None, n_classes])



# Convolution

def conv2d(name, x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    x = tf.nn.bias_add(x, b)

    #return tf.nn.relu(x, name=name)

    return tf.nn.elu(x, name=name)



# Pooling

def maxpool2d(name, x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)



weights = {

    'W1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),

    'W2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),

    'W4': tf.Variable(tf.truncated_normal([64 * 7 * 7, 784], stddev=0.1)),

    'Wo': tf.Variable(tf.truncated_normal([784, n_classes], stddev=0.1))

}



biases = {

    'b1': tf.Variable(tf.random_normal([32], stddev=0.1)),

    'b2': tf.Variable(tf.random_normal([64], stddev=0.1)),

    'b4': tf.Variable(tf.random_normal([784], stddev=0.1)),

    'bo': tf.Variable(tf.random_normal([n_classes], stddev=0.1))

}
def model(X, weights, biases):

    # Conv1

    x = tf.reshape(X, [-1, 28, 28, 1])

    

    # confine range to between 0 and 1

    x = x/255.

    

    #conv1 = tf.nn.relu(conv2d('conv1', x, weights['W1'], biases['b1']))

    conv1 = tf.nn.elu(conv2d('conv1', x, weights['W1'], biases['b1']))

    # Pool1

    pool1 = maxpool2d('pool1', conv1, k=2)



    # Conv2

    #conv2 = tf.nn.relu(conv2d('conv2', pool1, weights['W2'], biases['b2']))

    conv2 = tf.nn.relu(conv2d('conv2', pool1, weights['W2'], biases['b2']))



    # Pool2

    pool2 = maxpool2d('pool2', conv2, k=2)

    

    # Full connect layer

    fc = tf.reshape(pool2, [-1, weights['W4'].get_shape().as_list()[0]])

    fc = tf.add(tf.matmul(fc, weights['W4']), biases['b4'])

    #fc = tf.nn.relu(fc)

    fc = tf.nn.elu(fc)



    # output

    a = tf.add(tf.matmul(fc, weights['Wo']), biases['bo'])



    return a





# prediction

pred = model(X, weights, biases)



# cost

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

label = tf.argmax(pred, 1)

# evaluation

correct_pred = tf.equal(label, tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



init = tf.global_variables_initializer()

config = tf.ConfigProto()  

config.gpu_options.allow_growth=True  

with tf.Session(config=config) as sess:

    sess.run(init)  

    for e in range(epoch):

        step = 1

        while step*batch_size <= train_data.shape[0]:

            xs, ys = train_data[(step-1)*batch_size:step*batch_size, :], train_label[(step-1)*batch_size:step*batch_size, :]

            sess.run(optimizer, feed_dict={X:xs, y:ys})



            if step % 100 == 0:

                loss, acc = sess.run([cost, accuracy], feed_dict={X:xs, y:ys})



                print("Iter {0}, Minibatch Loss = {1}, Training accuracy = {2}".format(str(step),

                                                                                    loss, acc))

            step += 1

    print("Optimization Completed")

    test_labels = []

    for i in range(1000):

        xs, ys = test_data[i*28:(i+1)*28, :], test_data[i*28:(i+1)*28, 0:10]

        pred_ = sess.run(label, feed_dict={X:xs, y:ys})

        test_labels.extend(list(pred_))



f1 = open('label', 'wb')

pickle.dump(test_labels, f1)

f1.close()





# now we save the predictions to a csv ready for submission

df = pd.DataFrame({'Label': test_labels})

# Add 'ImageId' column

df1 = pd.concat([pd.Series(range(1,28001), name='ImageId'), 

                              df[['Label']]], axis=1)





df1.to_csv('ConvPool_X2.csv', index=False)

    