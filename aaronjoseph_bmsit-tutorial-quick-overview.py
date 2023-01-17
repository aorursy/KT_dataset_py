# A hashtag means comment in python

# Import the pandas library 

# And alias it as pd



##Type your code below this ( 1 line of code)

# Now we will see what all data files are present in the Docker Container

# os library help us go through the file system and will be used to list all the files present

import os

print(os.listdir("../input"))
print(os.listdir("../input/heart-disease-uci/"))
#This code now reads the files heart.csv and we have the data stored in the pandas variable heart_data

heart_data = pd.read_csv("../input/heart-disease-uci/heart.csv")
# Pandas Challenge ( Get first 5 rows of heart_data ) 1 line of code

# Pandas Challenge 1 line of code

# Pandas Challenge 1 line of code

# Pandas Challenge 1 line of code

# Pandas Challenge 1 line of code

# Pandas Challenge : Use pandas loc to print first column ('age') loc['age']



# Pandas Challenge : Use pandas iloc to print first column ('age') iloc[0]



# Pandas Challenge : Use pandas iloc/loc to print 3rd element of 'age' column

# Pandas Challenge 1 line of code

# Numpy challenge ( 1 line of code) Import the numpy library

# Numpy challenge ( 1 line of code) Create a numpy array with few elements (np.array([1,2,3])

# Also assign it to a variable say, a

a = np.array([1,2,3,4,5,6])

b = np.array([1])

# Numpy Challenge 1 line of code add a and b

a = np.array([1,2,3,4,5,6])

# Numpy Challenge 1 line of code find the square of a

import numpy as np 

import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import input_data # standard python class for downloading datasets

# read MNIST data

# https://stackoverflow.com/a/37540230/5411712

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_Data", one_hot=True)

print(mnist)
import tensorflow as tf
learning_rate = 0.01 # how fast to update weights; 0.01 is standard and pretty good

        # too big >> miss optimal soln; too small >> takes too long to find optimal soln

training_iteration = 30 # number of times to run the gradient descent (optimizer) step

batch_size = 100

display_step = 2
# TF graph input

x = tf.placeholder("float", [None, 784]) # mnist data image of shape; 28*28=784

     # notice images are 28px by 28px arrays & get "flattened" into 1D array of 784 pixels

y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition >> 10 classes to be "classified"



# create a model



# set model parameters

W = tf.Variable(tf.zeros([784, 10])) # weights (probabilities that affect how data flows in graph)

b = tf.Variable(tf.zeros([10]))      # biases (lets us shift the regression line to fit data)
# "scopes help us organize nodes in the graph visualizer called, Tensorboard"

with tf.name_scope("Wx_b") as scope:

    # First scope constructs a linear model (Logistic Regression)

    # `tf.nn` --- https://www.tensorflow.org/api_docs/python/tf/nn

    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax???? what about ReLU? Sigmoid? 

                                               # tf.nn.relu(biases=,features=,name=,weights=,x=)

                                               # tf.nn.softmax(_sentinel=,axis=,dim=,labels=,logits=,name=)
# Add summary operations to collect data

# helps us later visualize the distribution of the Weights and biases

# https://github.com/tensorflow/serving/issues/270

w_h = tf.summary.histogram("weights", W)

b_h = tf.summary.histogram("biases", b)
# More name scopes will clean up graph representation

with tf.name_scope("cost_function") as scope:

    # Second scope minimizes error using "cross entropy function" as the "cost function"

    # cross entropy function

    cost_function = -tf.reduce_sum(y*tf.log(model))

    # create a summary to monitor the cost function; for later visualization

    tf.summary.scalar("cost_function", cost_function)
with tf.name_scope("train") as scope:

    # Last scope Gradient Descent; the training algorithm

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
# initialize the variables

init = tf.initialize_all_variables()
# merge summaries into 1 operation

# https://github.com/tensorflow/tensorflow/issues/7737

merged_summary_op = tf.summary.merge_all()
print("learning_rate\t\t=\t" + str(learning_rate))

print("training_iteration\t=\t" + str(training_iteration))

print("batch_size\t\t=\t" + str(batch_size))

print("display_step\t\t=\t" + str(display_step))
# Start training by launching a session that executes the data flow graph

with tf.Session() as sess:

    sess.run(init)



    # Set the logs writer to the folder /tmp/tensorflow_logs

    # This is for all the visualizations later

    # https://stackoverflow.com/a/41483033/5411712

    summary_writer = tf.summary.FileWriter('./logs', graph_def=sess.graph_def)

    

    # Training cycle

    for i in range(training_iteration):

        avg_cost = 0.0 # prints out periodically to make sure model is "improving" ... goal is to minimize cost

        total_batch = int(mnist.train.num_examples / batch_size)

        # loop over all batches

        for b in range(total_batch): # for each example in training data

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # fit training using batch data

            # `optimizer` is Gradient Descent; used for 'backpropagation'

            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})

            # compute the average loss

            avg_cost += sess.run(cost_function, feed_dict={x:batch_xs, y:batch_ys})/total_batch

            # write logs for each iteration

            summary_str = sess.run(merged_summary_op, feed_dict={x:batch_xs, y:batch_ys})

            summary_writer.add_summary(summary_str, i * total_batch + b)

                                            # why `i * total_batch + b` ??? idk.

        # Display logs per iteration step

        if (i % display_step == 0):

            print("iteration:", '%04d' % (i+1), "avg_cost=", "{:9f}".format(avg_cost))



    print("\nTraining completed!\n")



    # Test the model

    # remember 'y' is the prediction variable

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

    # Calculate accuracy

    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
# optionally run the command in the notebook itself by uncommenting the line below

#!tensorboard --logdir=./logs