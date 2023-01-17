import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt
# your code here
# inputs provided as numpy array - these can be fed into placeholders using the feed_dict parameter

input1 = np.array([[2, 4],[7, 1]])

input2 = np.array([[3, 1],[5, 0]])



# your code here
# inputs provided as numpy array - these can be fed into placeholders using the feed_dict parameter

input1 = np.array([[3, 1, 7],[7, 12, 8]])

input2 = np.array([[1, 4],[0, 5], [6, 8]])



#myour code here
x = tf.placeholder(dtype=tf.float32)

y = tf.placeholder(dtype=tf.float32)

f = x*x + 5*x + y*y - 6*y + 50
xs = np.linspace(-10,10,21)

ys = np.linspace(-10,10,21)



with tf.Session() as sess:

    zv = np.array([[f.eval({x: xi, y: yj})for xi in xs] for yj in ys])



xv, yv = np.meshgrid(xs, ys)

plt.contour(xs, ys, zv, 50)

x_start = 7.5; y_start = -2.5

plt.plot(x_start, y_start, marker='o', markersize=5, color="red");
# your code here
# Assuming x_min and y_min are the values derived from gradient descent.

xv, yv = np.meshgrid(xs, ys)

plt.contour(xs, ys, zv, 50)

plt.plot(x_min, y_min, marker='x', markersize=10, color="blue");