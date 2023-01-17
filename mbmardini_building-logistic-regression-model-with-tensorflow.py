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
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set()
data = pd.read_csv('../input/Iris.csv')
data.head()
sns.pairplot(data.drop('Id', axis=1), hue='Species')
# let's see how many differnt type of spcies we have
data['Species'].unique()
data["Species"] = data["Species"].map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
# data.info()
data.head()
# let's check if there is any missing data
sns.heatmap(data.isnull(), cmap='plasma', cbar=False, yticklabels=False)
#Nope
data.shape
# As we said earlier we will work with just two classes. The first 50 samples are 'setosa', 
# second 50 are 'versicolor' and last 50 samples are 'virginica' 
data = data[:100]
import tensorflow as tf
input_size = 4  # Size of feature vector
output_size = 1 # Target vector
inputs = tf.placeholder(tf.float32,[None, input_size])
targets = tf.placeholder(tf.float32,[None, output_size])

weights = tf.Variable(tf.random_uniform([input_size, output_size], minval=-.1, maxval=.1))
biases = tf.Variable(tf.random_uniform([output_size], minval=-.1, maxval=.1))
outputs = tf.matmul(inputs, weights) + biases
# Please go and check cost function somewhere in case you don't recognize following 
# cost function
cost = tf.reduce_mean(tf.multiply(tf.sigmoid(outputs), targets))

# we will use pre-made Gradient descent to optimize or cost function
optimize = tf.train.GradientDescentOptimizer(learning_rate=.05).minimize(cost)
# So far nothing has been executed. In order to execute something in tenserflow 
# We need create a InteractiveSession and use run function to run what we have declared
sess = tf.InteractiveSession()

#Initialize variables
initializer = tf.global_variables_initializer()
sess.run(initializer)
# Learning
X = data.drop(['Id', 'Species'], axis=1).values
y = data['Species'].values
y = y.reshape((y.size,1))
# np.savez('iris', inputs=data.drop(['Id', 'Species'], axis=1).values, targets= data['Species'].values.reshape((y.size,1)))
# training_data = np.load('iris.npz')


# [optimize, mean_loss] are the funtion we want to run and they will return something
# optimize returns None that is why we assigned it to _ 
# mean_loss returns the error in our data with current weights and bias parameters
for e in range(100):
    _, curr_loss = sess.run([optimize, cost], feed_dict={inputs:X, targets:y})
    print(curr_loss)
sess.close()
