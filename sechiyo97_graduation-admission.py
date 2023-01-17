# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
%matplotlib inline
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('precision', 4)
np.set_printoptions(precision=3)

from mlxtend.preprocessing import minmax_scaling
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-paper')

tf.set_random_seed(777)  # for reproducibility

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Admission_Predict.csv", index_col=0)
df = df.dropna(axis=0)
numpy_matrix = df.as_matrix()
numpy_matrix = np.array(numpy_matrix)
x_data = numpy_matrix[:, 0:-1]
y_data = numpy_matrix[:, [-1]]
x_data = minmax_scaling(x_data, columns=[0,1,2,3,4,5,6])
x_data = normalize(x_data)
# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
my_data = x_data[0]
my_data = np.asarray([my_data.tolist()])
my_data
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([7, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.53)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
  
for step in range(4800):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val)

# Ask my score(used the first row of Admission_Predict)
print("Predicted: ", sess.run(
    hypothesis, feed_dict={X: my_data.tolist()})[0][0])
