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
a = tf.constant(3)

b = tf.constant(5)
with tf.Session() as sess:

    print ("a: i%" % sess.run(a), "b: i%" % sess.run(b))

    print ("Addition with constant: %i" % sess.run(a+b))

    print ("Multiplication with constant: %i" % sess.run(a*b))
a = tf.placeholder(tf.int16)

b = tf.placeholder(tf.int16)
add = tf.add(a,b)

mul = tf.multiply(a,b)
with tf.Session() as sess:

    print ("Addition with variables: %i" % sess.run (add, ))