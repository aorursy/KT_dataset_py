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
import tensorflow as tf
# declaring constant using tenserflow
hello = tf.constant('Hello')
type(hello)
world = tf.constant('World')
result = hello + world
result
# create session to run your tensor 
with tf.Session() as sess:
    result = sess.run(hello+world)
#Now print result 
result
import numpy as np
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
x_data
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(x_data,y_label,'*')
np.random.rand(2)
m = tf.Variable(0.2)
b = tf.Variable(0.98)
error = 0

for x,y in zip(x_data,y_label):
    
    y_hat = m*x + b  #Our predicted value
    
    error += (y-y_hat)**2 # The cost we want to minimize (we'll need to use an optimization function for the minimization!)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 100
    
    for i in range(epochs):
        
        sess.run(train)
        

    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])
final_slope
final_intercept
x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')

plt.plot(x_data,y_label,'*')
