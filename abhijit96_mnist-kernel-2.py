import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/train.csv')
label = data.label
data = data.drop('label', axis = 1)
X, X_test, Y, Y_test = train_test_split(data, label, test_size = 0.2)
with tf.Session() as sess:
    Y = tf.one_hot(Y,10)
    Y = sess.run(Y)
Y_test = Y_test.values
x = tf.placeholder(dtype = tf.float32, shape=[None,784])
y = tf.placeholder(dtype = tf.float32, shape=[None,10])
f1= tf.Variable(tf.random_normal(shape=[3,3,1,16]))
f2= tf.Variable(tf.random_normal(shape=[3,3,16,32]))
b1= tf.Variable(tf.random_normal(shape=[28,28,16]))
b2= tf.Variable(tf.random_normal(shape=[14,14,32]))
w1 = tf.Variable(tf.random_normal(shape=[1568,1000]))
b3= tf.Variable(tf.random_normal(shape=[1000]))
w2 = tf.Variable(tf.random_normal(shape=[1000,10]))
b4= tf.Variable(tf.random_normal(shape=[10]))

f1_1= tf.Variable(tf.random_normal(shape=[3,3,16,16]))
b1_1= tf.Variable(tf.random_normal(shape=[28,28,16]))

def cnn(x):
    l = tf.reshape(x, [-1,28,28,1])
    
    l = tf.nn.conv2d(l, f1, strides = [1,1,1,1], padding= 'SAME')
    l = tf.nn.relu(l + b1)
    
    l = tf.nn.conv2d(l, f1_1, strides = [1,1,1,1], padding= 'SAME')
    l = tf.nn.relu(l + b1_1)
    
    l = tf.nn.max_pool(l, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    l = tf.nn.conv2d(l, f2, strides = [1,1,1,1], padding= 'SAME')
    l = tf.nn.relu(l + b2)
    l = tf.nn.max_pool(l, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    l = tf.reshape(l,[-1,1568])
    
    l = tf.matmul(l,w1)
    l = tf.nn.relu(l + b3)
    
    l = tf.matmul(l,w2)
    l = l + b4
    
#     l = tf.nn.softmax(l)
    return l
ep = 50
pred = cnn(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = pred))#tf.square(tf.subtract(pred, y)))
optimizer = tf.train.AdamOptimizer().minimize(cost)
sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
for i in range(ep):
    k = 0
    epc = 0
    size = 100
    for _ in range(336):
        _, c = sess.run([optimizer, cost], feed_dict = {x: X[k:k+size], y: Y[k:k+size]})
        k = k+100
        epc += c
    print('Epoch: ',i+1,'   Cost: ', epc)
    p = tf.nn.softmax(pred)
    p = tf.argmax(p, axis =1)
    p = sess.run(p, feed_dict = {x: X_test})
    tr = 0
    for j in range(8400):
        if(p[j] == Y_test[j]):
            tr += 1
    print('Accuracy: ', tr/8400)
    print('Correct: ', tr)
data = pd.read_csv('../input/test.csv')
p1 = tf.argmax(pred, axis = 1)
p1 = sess.run(p1, feed_dict = {x: data})
sess.close()
d = pd.read_csv('../input/sample_submission.csv')
d.head()
res = pd.DataFrame(p1, columns = ['Label'])
res.index.names = ['ImageId']
res.index += 1
res.head()
res.to_csv('submit.csv', index_label = 'ImageId')

