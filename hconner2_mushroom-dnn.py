import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
data = pd.read_csv('../input/mushrooms.csv')

data.replace('?', np.nan, inplace=True)

data.dropna(inplace=True)

for column in data.columns:

    for var, i in zip(data[column].unique(), range(len(data[column].unique()))):

        data[column].replace(var, i, inplace=True)  

data.head()
data = np.array(data)



features = data[:, 1:].astype(np.float64)

features = StandardScaler().fit_transform(features)



labels = data[:, 0].astype(np.float64)
n_inputs = 22

n_hidden1 = 100

# n_hidden2 = 50

n_outputs = 2

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs))

y = tf.placeholder(tf.int64, shape=(None))
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)

# hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)

logits = tf.layers.dense(hidden1, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)
learning_rate = 0.01



optimizer = tf.train.GradientDescentOptimizer(learning_rate)

training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
n_epochs = 1000

batch_size = 2000
def next_batch(num, data, labels):

    idx = np.arange(0 , len(data))

    np.random.shuffle(idx)

    idx = idx[:num]

    data_shuffle = [data[i] for i in idx]

    labels_shuffle = [labels[i] for i in idx]



    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.5)
with tf.Session() as sess:

    sess.run(init)

    for epoch in range(1, n_epochs + 1):

        for i in range(len(data) // batch_size):

            X_batch_train, y_batch_train = next_batch(batch_size, X_train, y_train)

            X_batch_test, y_batch_test = next_batch(batch_size, X_test, y_test)

            sess.run(training_op, feed_dict={X: X_batch_train, y: y_batch_train})

#         predictions = tf.argmax(logits,1).eval(feed_dict={X: X_batch_test, y: y_batch_test})

#         roc_auc = tf.contrib.metrics.streaming_auc(predictions, y_batch_test)

        acc_train = accuracy.eval(feed_dict={X: X_batch_train, y: y_batch_train})

        acc_test = accuracy.eval(feed_dict={X: X_batch_test, y: y_batch_test})

        if epoch % 100 == 0:

            print('Epoch {}: Train Accuracy = {}%, Test Accuracy = {}%'.format(epoch, np.round(acc_train*100, 2), np.round(acc_test*100),2))