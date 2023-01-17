import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn import model_selection
import math
from mpl_toolkits.mplot3d import Axes3D

def plot3D(x,y,z, color='r', label=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(x, y, z, color=color)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    if label != None:
        ax.text2D(0.05, 0.95, label, transform=ax.transAxes)
f = lambda x1, x2: math.sin(x1)*math.cos(x2)
x1 = x2 = np.linspace(-3, 3, 30)
x = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
y = np.vectorize(f)(x[:, 0], x[:, 1])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

plot3D(x[:,0], x[:,1], y, 'r', label='Train data')
plot3D(x[:,0], x[:,1], y, 'g', label='Test data')
import tensorflow as tf
tf.set_random_seed(248)
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, 2], name='X_placeholder')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y_placeholder')
layer1 = tf.layers.dense(inputs=X, units=5, activation=tf.nn.tanh)
layer12 = tf.layers.dense(inputs=layer1, units=4, activation=tf.nn.tanh)
layer13 = tf.layers.dense(inputs=layer12, units=3, activation=tf.nn.tanh)
layer14 = tf.layers.dense(inputs=layer13, units=2, activation=tf.nn.tanh)
layer2 = tf.layers.dense(inputs=layer14, units=1)

loss = tf.losses.mean_squared_error(layer2, Y)
opt = tf.train.RMSPropOptimizer(0.001)
m_op = opt.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch_count = 5000
history_train_loss = []
history_test_loss = []

for i in range(epoch_count):
    ntw_out = sess.run([loss], feed_dict={X:x_test.reshape([len(x_test), -1]), Y:y_test.reshape([len(y_test), -1])})
    _, train_loss = sess.run([m_op, loss], feed_dict={X:x_train.reshape([len(x_train), -1]), Y:y_train.reshape([len(y_train), -1])})
    
    history_train_loss.append(np.mean(train_loss))
    history_test_loss.append(np.mean(ntw_out))
    
plt.plot(range(epoch_count), history_train_loss, 'r', label='Train loss')
plt.plot(range(epoch_count), history_test_loss, 'g', label='Test loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean Square Error')
plt.show()

network_output = sess.run([layer1, layer2], feed_dict={X:x_test.reshape([len(x_test), -1]), Y:y_test.reshape([len(y_test), -1])})

plot3D(x_test[:,0], x_test[:, 1], y_test, 'g', label='Ground truth surface')
plot3D(x_test[:,0], x_test[:, 1], network_output[1][:,0], 'y', label='Predicted surface')
_x1 = _x2 = np.arange(-10, 10, 1)
x_extrapolated = np.transpose([np.tile(_x1, len(_x2)), np.repeat(_x2, len(_x1))])
y = np.vectorize(f)(x_extrapolated[:, 0], x_extrapolated[:, 1])
network_output = sess.run([layer1, layer2], feed_dict={X:x_extrapolated.reshape([len(x_extrapolated), -1]), Y:y_test.reshape([len(y_test), -1])})

plot3D(x_extrapolated[:,0], x_extrapolated[:, 1], network_output[1][:,0], 'c', label='Extrapolated surface ground truth')
plot3D(x_extrapolated[:,0], x_extrapolated[:, 1], network_output[1][:,0], 'm', label='Extrapolated surface predicted')