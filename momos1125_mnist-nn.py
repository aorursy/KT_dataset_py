import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

x_train = train.drop(['label'], axis = 1)
#x_train.columns
y_train = train['label']
y_train = pd.get_dummies(y_train).as_matrix()
#y_train.head
x_train = x_train/255.0
test = test/255.0


data_size = x_train.shape[0]
validation_size = 2000
x_vali = x_train[:validation_size]
y_vali = y_train[:validation_size]
x_train = x_train[validation_size:]
y_train = y_train[validation_size:]

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
initializer=tf.contrib.layers.xavier_initializer()

W1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.01))
b1 = tf.Variable(tf.zeros([32]))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1+b1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob)
#######################################
W2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.01))
b2 = tf.Variable(tf.zeros([64]))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2+b2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob)
#######################################
W3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 256], stddev=0.01))
b3 = tf.Variable(tf.zeros([256]))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)+b3
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)
#######################################
W4 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.01))
b4 = tf.Variable(tf.zeros([10]))
model = tf.matmul(L3, W4)+b4


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(x_train.shape[0] / batch_size)
########
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#prediction = np.argmax(model, axis = 0)
########
for epoch in range(100):
    total_cost = 0

    for i in range(0,x_train.shape[0], batch_size):

        batch_xs = x_train[i:batch_size-1+i]
        batch_xs = batch_xs.values.reshape(-1,28,28,1)
        batch_ys = y_train[i:batch_size-1+i]

        _, cost_val = sess.run([optimizer, cost], 
                               feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch ),
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    print('Accuracy:', sess.run(accuracy,
                    feed_dict={X: x_vali.values.reshape(-1,28,28,1),
                                Y: y_vali,
                                keep_prob: 1}))

print('Done!')
print('Accuracy:', sess.run(accuracy,
                        feed_dict={X: x_vali.values.reshape(-1,28,28,1),
                                   Y: y_vali,
                                  keep_prob: 1}))
test_pred = sess.run(model,feed_dict={X: test.values.reshape(-1,28,28,1),keep_prob: 1})
sess.close()

test_pred_label = np.argmax(test_pred, axis = 1)

results = pd.DataFrame({'ImageID':pd.Series(range(1,28001)),
                        'Label':test_pred_label})

filename = 'Digit Recognition Predictions.csv'

results.to_csv(filename, index=False)
