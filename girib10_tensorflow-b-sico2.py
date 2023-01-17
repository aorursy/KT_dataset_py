import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.range(1,7,1)
x = tf.reshape(x,(2,3))
x.eval()
x.get_shape()
a = tf.Variable(tf.ones((2, 2)))
a
sess.run(tf.global_variables_initializer())
a.eval(session=sess)
sess.run(a.assign(tf.zeros((2,2))))
import numpy as np
import matplotlib.pyplot as plt
N = 100
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N, 1)
noise = np.random.normal(scale=noise_scale, size=(N, 1))
# Convert shape of y_np to (N,)
y_np = np.reshape(w_true * x_np + b_true + noise, (-1))
# Save image of the data distribution
plt.scatter(x_np, y_np)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 1)
plt.title("Toy Linear Regression Data, "r"$y = 5x + 2 + N(0, 1)$")
a = tf.placeholder(tf.float32, shape=(1,))
b = tf.placeholder(tf.float32, shape=(1,))
c = a + b
with tf.Session() as sess:
    c_eval = sess.run(c, {a: [1.], b: [2.]})
    print(c_eval)
