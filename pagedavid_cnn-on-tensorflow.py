import numpy as np
import pandas as pd
train = pd.read_csv('../input/emnist-balanced-train.csv', header=None)
test = pd.read_csv('../input/emnist-balanced-test.csv', header=None)
train.head()
train_data = train.iloc[:, 1:]
train_labels = train.iloc[:, 0]
test_data = test.iloc[:, 1:]
test_labels = test.iloc[:, 0]
train_labels = pd.get_dummies(train_labels)
test_labels = pd.get_dummies(test_labels)
train_labels.head()
train_data = train_data.values
train_labels = train_labels.values
test_data = test_data.values
test_labels = test_labels.values
del train, test
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(train_data[4].reshape([28, 28]), cmap='Greys_r')
plt.show()
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])
train_data = np.apply_along_axis(rotate, 1, train_data)
test_data = np.apply_along_axis(rotate, 1, test_data)
plt.imshow(train_data[4].reshape([28, 28]), cmap='Greys_r')
plt.show()
import tensorflow as tf
import tensorflow.contrib.slim as slim
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 47])
keep_prob = tf.placeholder(tf.float32)
with tf.device('/gpu:0'):
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

    net = slim.conv2d(x_image, 64, [5, 5])
    net = slim.max_pool2d(net, [2, 2])  # [-1, 14, 14, 64]

    net = slim.conv2d(net, 128, [5, 5])
    net = slim.max_pool2d(net, [2, 2])  # [-1, 7, 7, 128]

    x_flat = tf.reshape(net, [-1, 7 * 7 * 128])

    fc_1 = slim.fully_connected(x_flat, 1024)
    fc_2 = slim.fully_connected(fc_1, 128)
    fc_drop = tf.nn.dropout(fc_2, keep_prob)
    fc_3 = slim.fully_connected(fc_drop, 47, activation_fn=None)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=ys,
    logits=fc_3))
train = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct = tf.equal(tf.argmax(fc_3, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
NUM = 112800
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for i in range(int(NUM / 100)):
            x_batches, y_batches = train_data[i * 100: (i + 1) * 100], train_labels[i * 100: (i + 1) * 100]
            sess.run(train, feed_dict={xs: x_batches, ys: y_batches, keep_prob: 0.5})
            if i % 1000 == 0:
                acc = sess.run(accuracy, feed_dict={xs: test_data,
                                                    ys: test_labels,
                                                    keep_prob: 1.0})
                print(acc)
    acc = sess.run(accuracy, feed_dict={xs: test_data,
                                                ys: test_labels,
                                                keep_prob: 1.0})
    print(acc)