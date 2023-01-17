# Import required library

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

%matplotlib inline

style.use("ggplot")



print(tf.VERSION)



# Defining Inputs

x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="x")

w = tf.Variable(tf.ones((2, 1)), dtype=tf.float32)        

b = tf.Variable(tf.ones(1), dtype=tf.float32)              
# define operation for z

z = tf.add(tf.matmul(x, w), b)
# printing value of z for x = [[0.25, 0.15]]

init = tf.global_variables_initializer()

with tf.Session() as session:

    session.run(init)

    data = {x: [[0.25, 0.15]]}

    result_z = session.run(z, feed_dict= data)

    print("Value of z is {}".format(result_z[0][0]))
# using sigmoid as activation function 

out = tf.sigmoid(z)
# Executing the computation graph



with tf.Session() as session:

    session.run(init)

    result = session.run(out, feed_dict={x: [[0.25, 0.15]]})

    print(result[0][0])
def sigmoid_value(z):

    return (1 / (1 + np.exp(-z)))

sigmoid_value(result_z[0][0])
# generate data



data_points = np.linspace(-5, 5, 200)
# apply activation function on data_points

y_sigmoid = tf.sigmoid(data_points)
with tf.Session() as session:

    y_sigmoid = session.run(y_sigmoid)
# plot the Sigmoid Graph

plt.figure(figsize=(10, 6))

plt.plot(data_points, y_sigmoid, c="blue", label="sigmoid")

plt.ylim((-0.2, 1.2))

plt.legend(loc="best")

plt.show()
# using tanh as activation function

out_tanh = tf.tanh(z)

with tf.Session() as session:

    session.run(init)

    result = session.run(out_tanh, feed_dict={x: [[0.25, 0.15]]})

    print(result[0][0])
def tanh_value(z):

    return ((2 / (1 + np.exp(-2*z))) - 1)

tanh_value(result_z[0][0])
# apply activation function on data_points

y_tanh = tf.tanh(data_points)
with tf.Session() as session:

    y_tanh = session.run(y_tanh)
# plot the tanh Graph

plt.figure(figsize=(10, 6))

plt.plot(data_points, y_tanh, c="brown", label="tanh")

plt.ylim((-1.2, 1.2))

plt.legend(loc="best")

plt.show()
# using relu as activation function

out_relu = tf.nn.relu(z)

with tf.Session() as session:

    session.run(init)

    result = session.run(out_relu, feed_dict={x: [[0.25, 0.15]]})

    print(result[0][0])
def relu_value(z):

    if z < 0:

        return (0)

    else:

        return (z)

relu_value(result_z[0][0])
# apply activation function on data_points

y_relu = tf.nn.relu(data_points)
with tf.Session() as session:

    y_relu = session.run(y_relu)
# plot the relu graph

plt.figure(figsize=(10, 6))

plt.plot(data_points, y_relu, c="red", label="relu")

plt.ylim((-1, 5))

plt.legend(loc="best")

plt.show()
# using softplus as an activation function

out_softplus = tf.nn.softplus(z)

with tf.Session() as session:

    session.run(init)

    result = session.run(out_softplus, feed_dict={x: [[0.25, 0.15]]})

    print(result[0][0])
def softplus_value(z):

    return (np.log(1 + (np.exp(z))))

softplus_value(result_z[0][0])
# apply activation function on data_points

y_softplus = tf.nn.softplus(data_points)
with tf.Session() as session:

    y_softplus = session.run(y_softplus)
# plot the softplus graph

plt.figure(figsize=(10, 6))

plt.plot(data_points, y_softplus, c="green", label="softplus")

plt.ylim((-0.2, 6))

plt.legend(loc="best")

plt.show()