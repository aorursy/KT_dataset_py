import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
tf.reset_default_graph()

x = tf.Variable(3, name="x")

y = tf.Variable(4, name="y")

f = x*x*y + y + 2
# Initialize variables individually

sess = tf.Session()

sess.run(x.initializer)

sess.run(y.initializer)

result = sess.run(f)

sess.close()



print(result)
# Initialize all variables at once

sess = tf.Session()

sess.run(tf.global_variables_initializer())

result = sess.run(f)

sess.close()



print(result)
# Alternative way to create and run sessions

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    result = f.eval()

    

print(result)
tf.reset_default_graph()

x = tf.placeholder(tf.float32, name="x")

y = tf.placeholder(tf.float32, name="y")

f = x*x*y + y + 2
with tf.Session() as sess:    

    result1 = f.eval(feed_dict={x: 3, y: 4})

    result2 = f.eval(feed_dict={x: 5, y: 2})

    

print(result1)

print(result2)
grad = tf.gradients(f, [x,y])
with tf.Session() as sess:    

    df_by_dx = grad[0].eval(feed_dict={x: 3, y: 4})

    df_by_dy = grad[1].eval(feed_dict={x: 3, y: 4})

    

print(df_by_dx)

print(df_by_dy)