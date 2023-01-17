import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# prepare dataset
# linearly spaced and add a noice
X = np.linspace(0, 10, 10) - np.random.uniform(-1, 1, 10)
Y = np.linspace(0, 10, 10) - np.random.uniform(-1, 1, 10)
plt.plot(X, Y, '*')

# initialise the m and b with random variables
m = tf.Variable(1.01)
b = tf.Variable(1.01)
# Error function which needs to be minimized
error = 0

for x, y in zip(X, Y):
    Y_pred = m * x + b
    error += (y- Y_pred)**2 
# Gradient Decent Optimiser
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
model = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    training_steps = 10000
    
    for i in range(training_steps):
        sess.run(model)
        
    final_slobe, final_intercept = sess.run([m, b])
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slobe * x_test + final_intercept
plt.plot(x_test, y_pred_plot)
plt.plot(X, Y, '*')