import os, sys

import skimage.io
import skimage.transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.reset_default_graph()

input_tensor = tf.placeholder(tf.float32, [None, 112, 112, 3])
output_tensor = tf.placeholder(tf.float32, [None, 5])
keep_probability = tf.placeholder(tf.float32)

w_init = tf.random_normal_initializer(stddev=0.001)
b_init = tf.constant_initializer(0.001)

# Convalution layer 1
# input - [None,112,112,3], output - [None,56,56,32]
w1 = tf.get_variable('W1', shape=[3, 3, 3, 32], initializer=w_init)
b1 = tf.get_variable('b1', shape=[32], initializer=b_init)
conv_1 = tf.nn.conv2d(input_tensor, w1, strides=[1, 1, 1, 1], padding="SAME") + b1
activ_1 = tf.nn.relu(conv_1)
pool_1 = tf.nn.max_pool(activ_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Convalution layer
# input - [None,56,56,32], output - [None,28,28,64]
w2 = tf.get_variable('W2', shape=[3, 3, 32, 64], initializer=w_init)
b2 = tf.get_variable('b2', shape=[64], initializer=b_init)
conv_2 = tf.nn.conv2d(pool_1, w2, strides=[1, 1, 1, 1], padding="SAME") + b2
activ_2 = tf.nn.relu(conv_2)
pool_2 = tf.nn.max_pool(activ_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Convalution layer
# input - [None,28,28,64], output - [None,14,14,64]
w3 = tf.get_variable('W3', shape=[3, 3, 64, 64], initializer=w_init)
b3 = tf.get_variable('b3', shape=[64], initializer=b_init)
conv_3 = tf.nn.conv2d(pool_2, w3, strides=[1, 1, 1, 1], padding="SAME") + b3
activ_3 = tf.nn.relu(conv_3)
pool_3 = tf.nn.max_pool(activ_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Convalution layer
# input - [None,14,14,64], output - [None,7,7,32]
w4 = tf.get_variable('W4', shape=[3, 3, 64, 32], initializer=w_init)
b4 = tf.get_variable('b4', shape=[32], initializer=b_init)
conv_4 = tf.nn.conv2d(pool_3, w4, strides=[1, 1, 1, 1], padding="SAME") + b4
activ_4 = tf.nn.relu(conv_4)
pool_4 = tf.nn.max_pool(activ_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# input - [None,7,7,32], output - [None, 1568]
flatten_1 = tf.reshape(pool_4, shape=[-1, 7*7*32])

# Dence layer
# input - [None, 1568], output - [None, 1024]
w5 = tf.get_variable('W5', shape=[1568, 1024], initializer=w_init)
b5 = tf.get_variable('b5', shape=[1024], initializer=b_init)
activ_5 = tf.nn.relu(tf.matmul(flatten_1, w5)) + b5

# dropout
drop_1 = tf.nn.dropout(activ_5, keep_probability)

# Dence layer
# input - [None, 1024], output - [None, 5]
w6 = tf.get_variable('W6', shape=[1024, 5], initializer=w_init)
b6 = tf.get_variable('b6', shape=[5], initializer=b_init)
logit = tf.matmul(drop_1, w6) + b6

# softmax
output = tf.nn.softmax(logit)

# weigths adjustment
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=output_tensor))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# calc accuraty
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(output_tensor, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
generator = datagen.flow_from_directory(
    directory='../input/flowers/flowers/',
    target_size=(112, 112),
    batch_size=4323,
    class_mode='categorical', 
    shuffle=True)
X, y = next(generator)
X_train, y_train = X[:3500], y[:3500]
X_test, y_test = X[3500:], y[3500:]
train_loss_history = []; train_acc_history = []; 
test_loss_history = []; test_acc_history = []; 

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000): 
    train_indexes = np.random.choice(3500, 128)
    sess.run(
        train_step, 
        feed_dict={
            input_tensor: X_train[train_indexes], 
            output_tensor: y_train[train_indexes], 
            keep_probability: 0.35})

    if (i%100 == 0):
        acc_train, loss_train = sess.run(
            [accuracy, cross_entropy], 
            feed_dict={
            input_tensor: X_train[train_indexes], 
            output_tensor: y_train[train_indexes], 
            keep_probability: 1.0})
        
        train_loss_history.append(loss_train)
        train_acc_history.append(acc_train)
        
        test_indexes = np.random.choice(823, 128)
        acc_test, loss_test = sess.run(
            [accuracy, cross_entropy],
            feed_dict={
            input_tensor: X_test[test_indexes], 
            output_tensor: y_test[test_indexes], 
            keep_probability: 1.0})
        test_loss_history.append(loss_test)
        test_acc_history.append(acc_test)
        
        sys.stdout.write('\rstep: {0}, train loss: {1:.4}, train accuracy: {2:.5}{3:10}'.format(
            i, loss_test, acc_test, ' '))
fig, ax = plt.subplots(1,2,figsize=(20, 5))
ax[0].set_title('accuracy')
ax[0].plot(train_acc_history, label='train')
ax[0].plot(test_acc_history, label='test')
ax[0].legend()
ax[1].set_title('loss')
ax[1].plot(train_loss_history, label='train')
ax[1].plot(test_loss_history, label='test')
ax[1].legend()
w = sess.run(
    [w1, w2, w3, w4, w5, w6],
    feed_dict={
    input_tensor: X_test[test_indexes], 
    output_tensor: y_test[test_indexes], 
    keep_probability: 1.0})

for i in range(6):
    print('mean w{0}: {1:.5}'.format(i+1, w[i].mean()))
b = sess.run(
    [b1, b2, b3, b4, b5, b6],
    feed_dict={
    input_tensor: X_test[test_indexes], 
    output_tensor: y_test[test_indexes], 
    keep_probability: 1.0})

for i in range(6):
    print('mean b{0}: {1:.5}'.format(i+1, b[i].mean()))
result = sess.run(
    [conv_1, conv_2, conv_3,conv_4],
    feed_dict={
    input_tensor: X_test[test_indexes], 
    output_tensor: y_test[test_indexes], 
    keep_probability: 1.0})

for i in range(10):
    fig, ax = plt.subplots(1,5, figsize=(20,5))
    ax[0].imshow(X_test[test_indexes][i])
    ax[1].imshow(result[0][i].mean(axis=2))
    ax[2].imshow(result[1][i].mean(axis=2))
    ax[3].imshow(result[2][i].mean(axis=2))
    ax[4].imshow(result[3][i].mean(axis=2))
    plt.show()