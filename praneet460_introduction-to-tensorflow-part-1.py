import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore') # never print matching warnings

print("Tensorflow version ", tf.__version__)

print("Numpy Version ", np.__version__)
# Create nodes in a graph

a = tf.constant(value = 15, name="a", dtype = tf.int16)

b = tf.constant(value = 61, name="b", dtype = tf.int16)



# add them

c = tf.add(a, b, name="c")

print(c)
with tf.Session() as session:

    print (session.run(c))
# subtract them

z = tf.subtract(a, b, name="z")

with tf.Session() as session:

    print(session.run(z))
# multiply them

y = tf.multiply(a, b, name="y")

with tf.Session() as session:

    print(session.run(y))
# divide them

x = tf.divide(a, b, name="x")

with tf.Session() as session:

    print(session.run(x))
# define inputs

a = tf.placeholder(dtype = tf.float32)

b = tf.placeholder(dtype = tf.float32)



# c = a+b

c = tf.add(a, b, name="c")

# d = b-1

d = tf.subtract(b, 1, name="d")

# e = c*d

e = tf.multiply(c, d, name="e")



with tf.Session() as session:

    print(session.run(e, feed_dict={a:2.0, b:4.0}))