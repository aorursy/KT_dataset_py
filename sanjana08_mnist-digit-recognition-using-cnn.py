#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
train.head()
print(train.shape, test.shape)
train_x = np.asarray(train.iloc[:, 1:]).reshape(-1,28,28,1)
train_y = np.asarray(pd.get_dummies(train['label'], prefix='label'))
train_x.shape, train_y.shape
test_x = np.asarray(test.iloc[:,1:]).reshape(-1,28,28,1)
test_y = np.asarray(pd.get_dummies(test['label'], prefix='label'))
test_x.shape, test_y.shape
train_x = train_x/255
test_x = test_x/255
rows = 4
cols = 4
f = plt.figure(figsize=(rows,cols))
for i in range(rows*cols):
    f.add_subplot(rows,cols,i+1)
    plt.imshow(train_x[i].reshape([28,28]))
tf.reset_default_graph()
def init_weights(shape):
    init_w = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init_w)
def init_bias(shape):
    init_b = tf.constant(0.1, shape=shape)
    return tf.Variable(init_b)
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
def convolutional_layer(x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(x,W)+b)
def fully_connected_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer,W)+b
def next_batch(j, batch_size):
    x = train_x[j:j+batch_size].reshape(50,28,28,1)
    y = train_y[j:j+batch_size]
    j = (j + batch_size) % len(train_x)
    return x, y,j
x = tf.placeholder(tf.float32, shape=[None,28,28,1])
y_true = tf.placeholder(tf.float32, shape=[None,10])
x_input = tf.reshape(x,[-1,28,28,1])
conv_1 = convolutional_layer(x_input, shape=[5,5,1,32])
conv_1_pool = max_pool_2by2(conv_1)
conv_2 = convolutional_layer(conv_1_pool, shape=[5,5,32,64])
conv_2_pool = max_pool_2by2(conv_2)
conv_2_flat = tf.reshape(conv_2_pool, shape=[-1,7*7*64])
fully_connected = tf.nn.relu(fully_connected_layer(conv_2_flat, 1024))
hold_prob = tf.placeholder(tf.float32)
fully_connected_dropout = tf.nn.dropout(fully_connected, hold_prob)
y_pred = fully_connected_layer(fully_connected_dropout,10)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
steps = 5000
j = 0

with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x, batch_y, j = next_batch(j, 50)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})
        
        if i%100 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={x:test_x, y_true:test_y, hold_prob:1.0}))
            print('\n')
