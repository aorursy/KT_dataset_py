import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



df = pd.read_csv('../input/train.csv')

y = np.array([(np.arange(0,10,1) == label).astype(np.float32) for label in df.iloc[:, 0].values])

X = df.iloc[:, 1:].values

print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



stdsc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_std = stdsc.fit_transform(X_train.astype(np.float32))

X_test_std = stdsc.transform(X_test.astype(np.float32))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train_std.mean(), X_test_std.mean())

X_train_std = X_train_std.reshape(-1, 28, 28, 1)

X_test_std = X_test_std.reshape(-1, 28, 28, 1)

print(X_train_std.shape, X_test_std.shape, y_train.shape, y_test.shape)



del X_train, X_test, df, X, y
def accuracy(predictions, labels):

    return np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0] * 100
batch_size = 16

patch_size_1 = 5

patch_size_2 = 5

depth = 16

num_hidden = 32

image_size = 28

num_channels = 1

num_label = 10



graph = tf.Graph()



with graph.as_default():

    

#     global_step = tf.Variable(0, trainable=False)

#     learning_rate = 0.1

#     k = 0.5

#     learning_rate = tf.train.natural_exp_decay(learning_rate, global_step, 1000, k)

    

    tf_X_train = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

    tf_y_train = tf.placeholder(tf.float32, shape=(batch_size, num_label))

    tf_X = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))

    tf_y = tf.placeholder(tf.float32, shape=(None, num_label))

    

    W_1 = tf.Variable(tf.truncated_normal([patch_size_1, patch_size_1, num_channels, depth], stddev=0.1))

    b_1 = tf.Variable(tf.zeros([depth]))

    W_2 = tf.Variable(tf.truncated_normal([patch_size_2, patch_size_2, depth, depth], stddev=0.1))

    b_2 = tf.Variable(tf.truncated_normal([depth]))

    W_3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))

    b_3 = tf.Variable(tf.truncated_normal([num_hidden], stddev=0.1))

    W_4 = tf.Variable(tf.truncated_normal([num_hidden, num_label], stddev=0.1))

    b_4 = tf.Variable(tf.truncated_normal([num_label], stddev=0.1))

    

    def train_model(X):

        conv = tf.nn.conv2d(X, W_1, [1, 1, 1, 1], padding='SAME')

        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden = tf.nn.relu(conv + b_1)

        conv = tf.nn.conv2d(hidden, W_2, [1, 1, 1, 1], padding='SAME')

        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden = tf.nn.relu(conv + b_2)

        shape = hidden.get_shape().as_list()

        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, W_3) + b_3), 0.5)

        return tf.matmul(hidden, W_4) + b_4

    

    def predict_model(X):

        conv = tf.nn.conv2d(X, W_1, [1, 1, 1, 1], padding='SAME')

        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden = tf.nn.relu(conv + b_1)

        conv = tf.nn.conv2d(hidden, W_2, [1, 1, 1, 1], padding='SAME')

        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden = tf.nn.relu(conv + b_2)

        shape = hidden.get_shape().as_list()

        reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, W_3) + b_3)

        return tf.matmul(hidden, W_4) + b_4

    

    logits = train_model(tf_X_train)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_y_train))

    

    optimizer = tf.train.AdamOptimizer().minimize(loss)



    predict = tf.nn.softmax(predict_model(tf_X))

    train_prediction = tf.nn.softmax(predict_model(tf_X_train))

    

    saver = tf.train.Saver()
num_steps = 1000 * 20 + 1



with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()

    print('Initialized')

    for step in range(num_steps):

        offset = (step * batch_size) % (y_train.shape[0] - batch_size)

        batch_data = X_train_std[offset:(offset + batch_size), :, :, :]

        batch_labels = y_train[offset:(offset + batch_size), :]

        feed_dict = {tf_X_train : batch_data, tf_y_train : batch_labels}

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 1000 == 0):

            print('Minibatch loss at step %d: %f' % (step, l))

            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

    val_predict = session.run(predict, feed_dict={tf_X: X_test_std})

    print('Validation accuracy: %.1f%%' % accuracy(val_predict, y_test))