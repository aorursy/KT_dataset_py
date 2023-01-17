import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
observaions = 1000

X = np.random.uniform(-10,10, size= (observaions,1))

Z = np.random.uniform(-10,10, size= (observaions,1))
inputs = np.column_stack((X,Z))
inputs.shape
noise = np.random.uniform(-1,1,(observaions,1))

target = 2*X-3*Z+5+noise
init_range= 0.1

weight = np.random.uniform(-init_range, init_range,size = (2,1))

bias = np.random.uniform(-init_range, init_range, (1,1))

print(weight)

print(bias)
learning_rate = 0.05

c = []

for i in range(100):

    y_pred = np.dot(inputs,weight) + bias

    delta = y_pred - target

    cost = np.sum(delta**2)/(2*observaions)

    c.append(cost)

    print(cost)

    delta_sc = delta/observaions

    weight = weight - learning_rate*np.dot(inputs.T,delta_sc)

    bias = bias - learning_rate*np.sum(delta_sc)

Xaxis = np.arange(0,100)

plt.plot(Xaxis, c)

plt.xlabel('No. of Iterations')

plt.ylabel('Cost: (y-y_predicted)^2')
import tensorflow as tf

obser = 1000

x = np.random.uniform(-10,10,(obser,1))

z = np.random.uniform(-10,10,(obser,1))

gen_input = np.column_stack((x,z))

gen_output = 2*x-3*z+5

np.savez('TF_intro', inputs = gen_input, outputs = gen_output)
input_size=  2

output_size= 1

inputs = tf.placeholder(tf.float32, [None,input_size])

targets = tf.placeholder(tf.float32,[None, output_size])

weight = tf.Variable(tf.random_uniform([input_size, output_size], minval = -0.1, maxval = 0.1))

bias = tf.Variable(tf.random_uniform([output_size], minval = -0.1, maxval = 0.1))

outputs= tf.matmul(inputs,weight)+bias

mean_loss = tf.losses.mean_squared_error(labels=targets,predictions=outputs)/2.

optimize = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)

sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

sess.run(initializer)

training_data = np.load('TF_intro.npz')

c = []

for e in range(100):

    _,cost = sess.run([optimize, mean_loss], feed_dict={inputs: training_data['inputs'], targets: training_data['outputs']})

    print('cost:', cost)

    c.append(cost)
plt.plot(Xaxis, c)

plt.xlabel('No. of Epochs')

plt.ylabel('Cost: (y-y_predicted)^2')