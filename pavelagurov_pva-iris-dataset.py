# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataSet = pd.read_csv('../input/Iris.csv')
dataSet = dataSet.drop(columns=['Id'])
print(dataSet.head())

trainDataSet, testDataSet = train_test_split(dataSet)
print(len(trainDataSet), len(testDataSet))
x_train = trainDataSet.iloc[:,:-1] # features
y_train = trainDataSet.iloc[:,-1: ] # lables
print(x_train.head())
print(y_train.head())

x_test = testDataSet.iloc[:,:-1] # features
y_test = testDataSet.iloc[:,-1: ] # lables
OHE = OneHotEncoder(categories='auto')
OHE.fit(y_train)
y_train = OHE.transform(y_train).toarray()
print(y_train[0])
y_test = OHE.transform(y_test).toarray()
print(y_test[0])
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (correctly_predicted * 100) / predictions.shape[0]
    return accu
features = tf.placeholder(dtype = tf.float32, shape=[None, 4])

hidden_dim = 10

layer1_weights = tf.Variable(tf.random_normal([4,hidden_dim]))
layer1_bias = tf.Variable(tf.random_normal([hidden_dim]))
# [None, 4] x [4,hidden_dim] = [None, hidden_dim]
hidden1 = tf.nn.tanh(tf.add(tf.matmul(features, layer1_weights), layer1_bias))

output_layer = tf.Variable(tf.random_normal([hidden_dim,3]))
output_bias = tf.Variable(tf.random_normal([3]))
# [None, hidden_dim] x [hidden_dim, 3] = [None, 3]
output = tf.add(tf.matmul(hidden1, output_layer), output_bias)
target = tf.placeholder(dtype = tf.float32, shape=[None, 3])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = target))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)

train = optimizer.minimize(loss)
train_pred = tf.nn.softmax(output)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(15000):
    _, epoch_loss, epoch_predictions, _ = sess.run([train, loss, train_pred, output], feed_dict={features: x_train, target: y_train})
    epoch_accuracy = accuracy(epoch_predictions, y_train)
    if epoch % 500 == 0:
        print("Epoch #%s, Loss: %s, Accuracy: %s" %(epoch, epoch_loss, epoch_accuracy))