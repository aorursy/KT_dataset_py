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
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
X_test = test_df.values
import tensorflow as tf
X = train_df.drop('label',axis=1)
Y = train_df['label']
Y = pd.get_dummies(Y)
X = X.values
Y = Y.values
n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batches = int(X.shape[0]/batch_size)
        for epoch in range(hm_epochs):
            epoch_loss=0
            
            for i in range(total_batches):
                epoch_x = X[i*100:(i+1)*100][:]
                epoch_y = Y[i*100:(i+1)*100][:]
                _, c = sess.run([optimizer,cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch:", epoch , 'completed out of ', hm_epochs,' loss :' ,epoch_loss)

        Pred = sess.run(prediction, feed_dict = {x:X_test})
    return Pred
Pred = train_neural_network(x)
A = tf.argmax(Pred,1)
sess = tf.Session()
a = sess.run(A)
a = a.reshape(1,28000)
df_output = pd.DataFrame(a)
df_output = df_output.T
df_output = df_output.reset_index()
df_output.columns = ['ImageId','Label']
df_output['ImageId'] = df_output['ImageId']+1
df_output.to_csv('submission2.csv', index=False)
