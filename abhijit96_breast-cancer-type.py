import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
d = pd.read_csv('../input/data.csv')
d = d.drop('Unnamed: 32', axis = 1)
d = d.drop('id', axis = 1)
le = LabelEncoder()
le.fit(d.diagnosis)
d.diagnosis = le.transform(d.diagnosis)
d = shuffle(d)

label = d.diagnosis
d = d.drop('diagnosis', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(d, label, test_size = 0.5)
x_train = x_train.values
x_test  = x_test.values
y_train = y_train.values
y_test  = y_test.values
w1 = tf.Variable(tf.random_normal(shape = [30, 100]))
w2 = tf.Variable(tf.random_normal(shape = [100, 100]))
w3 = tf.Variable(tf.random_normal(shape = [100, 1]))

b1 = tf.Variable(tf.random_normal(shape = [100]))
b2 = tf.Variable(tf.random_normal(shape = [100]))
b3 = tf.Variable(tf.random_normal(shape = [1]))
def nn(x):
    l = tf.matmul(x, w1)
    l = tf.nn.relu(l + b1)
    
#     l = tf.matmul(l, w2)
#     l = tf.nn.relu(l + b2)
    
    l = tf.matmul(l, w3)
    l = l + b3
    
    return l
x = tf.placeholder(dtype = tf.float32, shape = [None, 30])
y = tf.placeholder(dtype = tf.float32, shape = [None, 1])

t1 = tf.placeholder(dtype = tf.float32, shape = [None, 1])
t2 = tf.placeholder(dtype = tf.float32, shape = [None, 1])

pred = nn(x)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = pred))
opt  = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
y_train = sess.run(tf.reshape(y_train, [-1,1]))
ep = 10
for i in range(ep):
    _,c = sess.run([opt, cost], feed_dict = {x: x_train, y: y_train})
    print('Epoch: ', i+1, ' of ', ep, '   cost:  ', c)
    
    p = tf.nn.sigmoid(pred)
    p = sess.run(p, feed_dict = {x: x_test})
    
    r = []
    for j in p:
        if(j>=0.5):
            r.append(1)
        else:
            r.append(0)
    k = 0
    cnt = 0
    for j in range(len(r)):
        if(r[j] == p[j]):
            k += 1;
        cnt += 1
    print('Accuracy: ', "{0:.3f}".format(k/cnt * 100) , '%')
sess.close()