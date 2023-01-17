import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y= pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)
learning_rate = .01
epochs = 25
batch_size = 25
display_step = 1
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 4]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 3]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))
pred = tf.nn.softmax(tf.matmul(x,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(1000):
        avg_cost = 0
        total_batch = int(trainX.shape[0]/batch_size)
      # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = trainX[i*batch_size:(i+1)*batch_size],trainY[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
        if ((epoch+1) % 50 == 0):
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", sess.run(accuracy,feed_dict = {x: testX, y: testY}))
