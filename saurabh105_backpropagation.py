import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import tensorflow as tf
sess = tf.Session()
x_vals = np.random.normal(1, 0.1, 100) #mean=1, std=0.1
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.zeros([1,1]))
batch_size = 16
output = tf.add(tf.multiply(A, x_data), b)
loss = tf.reduce_mean(tf.square(output-y_target))
opt = tf.train.GradientDescentOptimizer(0.05)
train_step = opt.minimize(loss)
sess.run(tf.initialize_all_variables())
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    if (i)%5==0:
        print("#Step:", i)
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print("Loss= ", str(temp_loss))
        loss_batch.append(temp_loss)
    
loss_st = []
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = ([[x_vals[rand_index]]])
    rand_y = ([[y_vals[rand_index]]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    if (i)%5==0:
        print("#Step:", i)
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print("Loss= ", str(temp_loss))
        loss_st.append(temp_loss)
    
plt.plot(range(0, 100, 5), loss_st, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size=16')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()
