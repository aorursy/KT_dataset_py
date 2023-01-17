# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

from pandas import Series, DataFrame

import requests

from sklearn import datasets

from sklearn import preprocessing

from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/weatherAUS.csv')

data.head()
data.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm','RISK_MM','Date','Location'],axis=1,inplace=True)

data.dropna(inplace=True)

data[['RainTomorrow','RainToday']] = data[['RainTomorrow','RainToday']].replace({'No':0,'Yes':1})

print(data.info())

data.head()
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

data = pd.get_dummies(data, columns=categorical_columns)

x_vals = data.drop(['RainTomorrow'],axis=1)

y_vals = data['RainTomorrow']

x_vals.head()
from sklearn.model_selection import train_test_split

x_vals_train, x_vals_test, y_vals_train, y_vals_test = train_test_split(x_vals, y_vals, test_size=0.20, random_state=88)

x_vals_train.head()

#x_vals_test.head()
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(x_vals_train)

x_vals_train = np.array(pd.DataFrame(scaler.transform(x_vals_train), index=x_vals_train.index, columns=x_vals_train.columns))

x_vals_test = np.array(pd.DataFrame(scaler.transform(x_vals_test), index=x_vals_test.index, columns=x_vals_test.columns))

print(x_vals_train)
batch_size = 25

x_data = tf.placeholder(shape=[None, 61],dtype=tf.float32)

y_target = tf.placeholder(shape=[None, 1],dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[61,1]))

b = tf.Variable(tf.random_normal(shape=[1,1]))

#b = tf.Variable(tf.zeros(1))

model_output = tf.add(tf.matmul(x_data,A),b)



loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

init = tf.global_variables_initializer()

sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train_step = my_opt.minimize(loss)



prediction = tf.round(tf.sigmoid(model_output))

prediction_correct = tf.cast(tf.equal(prediction,y_target),tf.float32)

accuracy = tf.reduce_mean(prediction_correct)



loss_vec = []

train_acc = []

test_acc = []

for i in range(1500):

    rand_index = np.random.choice(len(x_vals_train),size=batch_size)

    rand_x = x_vals_train[rand_index]

    rand_y = np.transpose([y_vals_train[rand_index]])

    sess.run(train_step,feed_dict={x_data: rand_x,y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x,y_target: rand_y})

    loss_vec.append(temp_loss)

    temp_acc_train = sess.run(accuracy,feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})

    train_acc.append(temp_loss)

    temp_acc_test = sess.run(accuracy,feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})

    test_acc.append(temp_acc_test)

plt.plot(loss_vec,'k-')

plt.title('Cross Entropy Loss per Generation')

plt.xlabel('Generation')

plt.ylabel('Cross Entropy Loss')

plt.show()

plt.plot(train_acc,'k-',label='Train Set Accuracy')

plt.plot(test_acc,'r--',label='Test Set Accuracy')

plt.xlabel('Generation')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')

plt.show()