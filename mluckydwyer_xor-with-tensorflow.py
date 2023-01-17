import tensorflow as tf
#import os
#print(os.listdir("../input"))

# Training Epochs 100,000
Epochs = 100001

# Test/Train Data
XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0], [1], [1], [0]]

# Setting up Tensorflow vaiables (inputs, weights, and biases)
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

w1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Weight1")
w2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Weight2")

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")
# Building Model
layer1 = tf.sigmoid(tf.matmul(x_, w1) + b1)
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

# Set the cost function
cost = tf.reduce_mean(((y_ + tf.log(layer2)) + ((1 - y_) * tf.log(1.0 - layer2))) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# List Devices
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range (Epochs):
    error = sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    if i % 10000 == 0:
        print('\nEpoch: ' + str(i) + '\nCost: ' + str(sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y})))