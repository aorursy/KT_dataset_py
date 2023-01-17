import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
A = tf.placeholder(tf.int32, shape=(None, None),name="A")
B = tf.placeholder(tf.int32, shape=(None, None),name="B")
C = tf.matmul(A,B)
# inputs provided as numpy array - these can be fed into placeholders using the feed_dict parameter
input1 = np.array([[2, 4],[7, 1]])
input2 = np.array([[3, 1],[5, 0]])

with tf.Session() as sess:
    result = C.eval(feed_dict={A: input1,B: input2})
    
print(result)
# inputs provided as numpy array - these can be fed into placeholders using the feed_dict parameter
input1 = np.array([[3, 1, 7],[7, 12, 8]])
input2 = np.array([[1, 4],[0, 5], [6, 8]])

with tf.Session() as sess:
    result = C.eval(feed_dict={A: input1,B: input2})
    
print(result)
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
plt.hold()
x_t = x_start; y_t = y_start
learning_rate = 0.05
max_its = 100
convergence_threshold = 0.001
plt.contour(xs, ys, zv, 50)

grad = tf.gradients(f, [x,y])

with tf.Session() as sess:
    for i in range(max_its):
        grad_x = grad[0].eval({x: x_t, y: y_t})    
        grad_y = grad[1].eval({x: x_t, y: y_t})
        x_old = x_t; y_old = y_t
        x_t = x_t - learning_rate * grad_x        
        y_t = y_t - learning_rate * grad_y
        plt.plot(x_t, y_t, marker='o', markersize=5, color="red");
        if (abs(x_old - x_t) < convergence_threshold) & (abs(y_old - y_t) < convergence_threshold): break;
            
x_min = x_t; y_min = y_t
# Assuming x_min and y_min are the values derived from gradient descent.
xv, yv = np.meshgrid(xs, ys)
plt.contour(xs, ys, zv, 50)
plt.plot(x_min, y_min, marker='x', markersize=10, color="blue");
