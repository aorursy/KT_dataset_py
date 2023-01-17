import json

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import tensorflow as tf

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



print(train_data.shape)

print(test_data.shape)
f, axarr = plt.subplots(10, 10)

for row in range(10):

    for column in range(10):

        entry = train_data[train_data['label']==column].iloc[row].drop('label').as_matrix()

        axarr[row, column].imshow(entry.reshape([28, 28]))

        axarr[row, column].get_xaxis().set_visible(False)

        axarr[row, column].get_yaxis().set_visible(False)
# input tensor

x = tf.placeholder(tf.float32, [None, 784])



# weights (w) and biases (s) 

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))



# model output (y)

y = tf.nn.softmax(tf.matmul(x, W) + b)
# target

y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_val_ratio = 0.7

train_data_size = len(train_data)

train_set = train_data[:int(train_data_size*train_val_ratio)]

val_set = train_data[int(train_data_size*train_val_ratio)+1:]



init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()

sess.run(init)



train_eval_list = []

val_eval_list = []

for i in range(1000):

    batch = train_set.sample(frac=0.1)

    batch_xs = batch.drop('label', axis=1).as_matrix()/255.0

    batch_ys = pd.get_dummies(batch['label']).as_matrix()

    val_xs = val_set.drop('label', axis=1).as_matrix()/255.0

    val_ys = pd.get_dummies(val_set['label']).as_matrix()



    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

 

    train_eval = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

    val_eval = sess.run(accuracy, feed_dict={x: val_xs, y_: val_ys})

    

    train_eval_list.append(train_eval)

    val_eval_list.append(val_eval)



saver.save(sess, "logistic_regression.ckpt")

sess.close()
plt.plot(train_eval_list, label='Train set')

plt.plot(val_eval_list, label='Validation set')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc=4)
saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, "logistic_regression.ckpt")

    predict = sess.run(y, feed_dict={x: test_data.as_matrix() / 255.0})

    pred = [[i + 1, np.argmax(one_hot_list)] for i, one_hot_list in enumerate(predict)]

    submission = pd.DataFrame(pred, columns=['ImageId', 'Label'])

    submission.to_csv('submission_logreg.csv', index=False)