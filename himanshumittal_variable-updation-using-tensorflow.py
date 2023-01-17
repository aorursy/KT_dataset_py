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
# importing the tensorflow library
import tensorflow as tf
# Purpose: Defining variable and understanding session. Mainly about the importance of placement of statements...
#            outside the seesion

x = tf.Variable(0, name='x')
cons_y = tf.constant(1, name = 'constant_y')
x = x + cons_y
model = tf.global_variables_initializer()
with tf.Session() as sess:
    print("Initializing the variables")
    sess.run(model)
    print("Initial value of x:")
    print(sess.run(x))
    for i in range(5):
        update_x = sess.run(x)
        print("Updated value of x: %s" %update_x)
# Purpose: Defining variable and understanding session. Mainly about the importance of placement of statements...
#            inside the seesion

x = tf.Variable(0, name='x')
model = tf.global_variables_initializer()
with tf.Session() as sess:
    print("Initializing the variables")
    sess.run(model)
    print("Initial value of x:")
    print(sess.run(x))
    for i in range(5):
        x = x + 1
        update_x = sess.run(x + 1)
        print("Updated value of x: %s" %sess.run(x))
        print("Updated value of update_x: %s" %update_x)
# Purpose: Checking the updated value of x

with tf.Session() as sess:
    sess.run(model)
    y=sess.run(x)
    print("Current value of x: %s" %y)
# Purpose: Defining variable and placeholders

x = tf.Variable(0, name='x')
a = tf.placeholder(tf.int32)
update_x = x + a
model = tf.global_variables_initializer()
with tf.Session() as sess:
    print("Initializing the variables")
    sess.run(model)
    print("Initial value of x:")
    print(sess.run(x))
    for i in range(5):
        u_x = sess.run(update_x,feed_dict={a:i})
        print("Updated value of x: %s %s %s" %(u_x,sess.run(x),i))
