import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

print("Using tensorflow v" + tf.__version__)
data = pd.read_csv('../input/Iris.csv')

data.head()
# Describe dataset

data.describe()
data['Species'].unique()
class_dict ={

    'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2

}

data['Species']=data['Species'].apply(lambda x: class_dict[x])

data.describe()
train, test = train_test_split(data)

x_train = train.drop(['Species', 'Id'], axis=1)

y_train = train['Species'].values.astype('int32')

encoder = OneHotEncoder(sparse=False, categories='auto')

y_train = y_train = encoder.fit_transform(y_train.reshape(-1,1))

x_test = test.drop(['Species', 'Id'], axis=1)

y_test = test['Species']

print('X train = {}'.format(x_train.shape))

print('Y train = {}'.format(y_train.shape))

print('X test = {}'.format(x_test.shape))

print('Y test = {}'.format(y_test.shape))

weights = tf.Variable(tf.random_normal((4,3)), name="weights")

bias = tf.Variable(tf.random_normal([3]), name="bias")



with tf.name_scope('Train'):

    x_tr = tf.placeholder(tf.float32, shape=(None, 4), name="input")

    y_tr = tf.placeholder(tf.float32, shape=(None,3), name="output")



with tf.name_scope('Test'):

    x_te = tf.placeholder(tf.float32, shape=(None, 4), name="input")

    y_te = tf.placeholder(tf.int64, shape=(None,), name="output")

    

    logits=tf.add(tf.matmul(x_te, weights), bias)

    y_pred = tf.argmax(tf.nn.softmax(logits), axis=1)

    

    with tf.name_scope('Loss'):

        test_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y_te, depth=3), logits)

    with tf.name_scope('correct_prediction'):

        correct_prediction = tf.equal(y_te, y_pred)

    with tf.name_scope('accuracy'):

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_loss_summary = tf.summary.scalar('Loss', test_loss)

    





with tf.name_scope('Train'):

    y_h = tf.add(tf.matmul(x_tr, weights), bias)



    with tf.name_scope('Loss'):

        train_loss = tf.losses.softmax_cross_entropy(y_tr, y_h)

    train_loss_summary = tf.summary.scalar('Loss', train_loss)

    

    train_op = tf.train.AdamOptimizer().minimize(train_loss)



# For visualization using tensorboard



init_g = tf.global_variables_initializer()

init_l = tf.local_variables_initializer()



with tf.Session() as sess:

    train_writer = tf.summary.FileWriter('Visualization/Train', sess.graph)

    test_writer = tf.summary.FileWriter('Visualization/Test')

    sess.run([init_g, init_l])

    for epoch in range(1, 10001):

        _, l, summary= sess.run([train_op, train_loss, train_loss_summary], feed_dict={x_tr: x_train.values, y_tr: y_train})

        train_writer.add_summary(summary, epoch)

        

        acc1, test_summary = sess.run([accuracy, test_loss_summary], feed_dict={x_te: x_test.values, y_te: y_test})

        test_writer.add_summary(test_summary, epoch)

        if epoch % 2000 == 0:

            print("Epoch {}, loss={}, test accuracy={}".format(epoch, l, acc1))        