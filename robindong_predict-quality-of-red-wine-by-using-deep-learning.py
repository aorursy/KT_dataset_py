# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

wine = pd.read_csv('../input/winequality-red.csv')
x = wine.drop('quality', axis = 1)
y = wine['quality']

# Split data to 'train' and 'test'
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 23)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

sgdc = SGDClassifier(max_iter = 800)
sgdc.fit(x_train, y_train)
y_predict = sgdc.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(score)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 200)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(score)
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.preprocessing import LabelBinarizer

learning_rate = 0.001

nr_features = x_train.shape[1]
nr_samples = x_train.shape[0]
quality_rank = len(y_train.unique())

lb = LabelBinarizer()
label_train = lb.fit_transform(y_train)
label_test = lb.fit_transform(y_test)

# Build graph of neural network
with tf.device('/cpu:0'):
    x_holder = tf.placeholder(tf.float32, (None, nr_features))
    y_holder = tf.placeholder(tf.int32, (None, quality_rank))
    keep_prob = tf.placeholder(tf.float32)

    input = tf.nn.dropout(x_holder, keep_prob = keep_prob)
    
    layer1 = fully_connected(input, 1024)
    layer2 = fully_connected(layer1, 512)
    layer3 = fully_connected(layer2, 256)
    layer4 = fully_connected(layer3, 128)
    logits = fully_connected(layer4, quality_rank)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y_holder)
    loss = tf.reduce_sum(loss)
    opt = tf.train.AdamOptimizer(learning_rate)
    train_op = opt.minimize(loss)

    # evaluation
    correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_holder, 1))
    correct_predict = tf.cast(correct_predict, tf.float32)
    accuracy = tf.reduce_mean(correct_predict)
 
# Run graph (training)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        _, loss_ = sess.run([train_op, loss], feed_dict = {x_holder: x_train, y_holder: label_train, keep_prob: 0.9})
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict = {x_holder: x_test, y_holder: label_test, keep_prob: 1.0})
            print('loss: %g, accuracy: %g' % (loss_, acc))            