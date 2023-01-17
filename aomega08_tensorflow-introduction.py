# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)

node2 = tf.constant(4.0) # also tf.float32 implicitly

print(node1, node2)
sess = tf.Session()
print(sess.run([node1, node2]))
node3 = tf.add(node1, node2)

print(node3)

print(sess.run(node3))
a = tf.placeholder(tf.float32)

b = tf.placeholder(tf.float32)

adder_node = a + b # same as tf.add(a, b)
print(adder_node)

print(sess.run(adder_node, { a: 3, b: 4.5 }))

print(sess.run(adder_node, { a: [1, 3], b: [2, 4] }))
add_and_triple = adder_node * 3.

print(sess.run(add_and_triple, { a: 3, b: 4.5 }))
W = tf.Variable([.3], dtype=tf.float32)

b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W * x + b
init = tf.global_variables_initializer()

sess.run(init)
print(sess.run(linear_model, { x: [1, 2, 3, 4] }))
y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, { x: [1, 2, 3, 4], y: [0, -1, -2, -3] }))
fixW = tf.assign(W, [-1.])

fixb = tf.assign(b, [1.])

sess.run([fixW, fixb])

print(sess.run(loss, { x: [1, 2, 3, 4], y: [0, -1, -2, -3] }))
sess.run(init)
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)
for i in range(1000):

  sess.run(train, { x: [1, 2, 3, 4], y: [0, -1, -2, -3] })



print(sess.run([W, b]))
features = [tf.contrib.layers.real_valued_column('x', dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
x_train = np.array([1., 2., 3., 4.])

y_train = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,

                                              batch_size=4,

                                              num_epochs=1000)
x_eval = np.array([2., 5., 8., 1.])

y_eval = np.array([-1.01, -4.1, -7, 0.])

eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_eval}, y_eval,

                                                   batch_size=4,

                                                   num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=10000)
train_loss = estimator.evaluate(input_fn=input_fn)

print("train loss: %r"% train_loss)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)

print("eval loss: %r"% eval_loss)