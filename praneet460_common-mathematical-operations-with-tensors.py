import tensorflow as tf

import numpy as np

np.random.seed(0)

print("Tensorflow version ", tf.__version__)

print("Numpy version ", np.__version__)
a = tf.placeholder(dtype=tf.float32, name = "a")

b = tf.placeholder(dtype=tf.float32, name = "b")
# defining the session

sess = tf.InteractiveSession()
add = tf.add(a, b, name="add")

a1 = np.random.randint(low=1, high=100)

b1 = np.random.randint(low=1, high=100)

data = {a: a1, b: b1}

print("Sum of {} and {} is {}".format(a1, b1, add.eval(feed_dict = data)))
sub = tf.subtract(a, b, name="sub")

print("Subtraction of {} from {} is {}".format(a1, b1, sub.eval(feed_dict = data)))
mult = tf.multiply(a, b, name="mult")

print("Multiplication of {} and {} is {}".format(a1, b1, mult.eval(feed_dict = data)))
div = tf.divide(a, b, name="div")

print("Division of {} from {} is {}".format(a1, b1, div.eval(feed_dict = data)))
mod = tf.mod(a, b, name="mod")

print("Modulus of {} and {} is {}".format(a1, b1, mod.eval(feed_dict = data)))
ab = tf.abs(a, name="ab")

print("Absolute value of {} is {}".format(-a1, ab.eval(feed_dict={a: -a1})))
neg = tf.negative(a, name="neg")

print("Negative value of {} is {}".format(a1, neg.eval(feed_dict={a: a1})))
sign = tf.sign(a, name="sign")

print("Sign of {} is {}".format(a1, sign.eval(feed_dict={a: a1}))) # 1 if x>0; 0 if x=0; -1 if x<0
sqr = tf.square(a, name="sqr")

print("Square of {} is {}".format(a1, sqr.eval(feed_dict={a: a1})))
rund = tf.round(a, name="rund")

a2 = np.random.random_sample()

print("Round value of {} is {}".format(a2, rund.eval(feed_dict={a: a2})))
sqrt = tf.sqrt(a, name="sqrt")

print("Squre root of {} is {}".format(a1, sqrt.eval(feed_dict={a: a1})))
powr = tf.pow(a, b, name="powr")

a3 = np.random.randint(low=1, high=10)

b3 = np.random.randint(low=1, high=10)

data_powr = {a: a3, b: b3}

print("{} to the power {} is {}".format(a3, b3, powr.eval(feed_dict=data_powr)))
exp = tf.exp(a, name="exp")

print("Exponential of {} is {}".format(a3, exp.eval(feed_dict={a: a3})))
log = tf.log(a, name="log")

print("Log of {} to the base e is {}".format(a3, log.eval(feed_dict={a: a3})))
maxi = tf.maximum(a, b, name="maxi")

print("Maximum value between {} and {} is {}".format(a1, b1, maxi.eval(feed_dict=data)))
mini =tf.minimum(a, b, name="mini")

print("Minimum value between {} and {} is {}".format(a1, b1, mini.eval(feed_dict=data)))
cos = tf.cos(a, name="cos")

print("Cosine of {} is {}".format(a1, cos.eval(feed_dict={a: a1})))
sin = tf.sin(a, name="sin")

print("Sine of {} is {}".format(a1, sin.eval(feed_dict={a: a1})))