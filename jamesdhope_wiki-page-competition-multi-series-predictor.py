import tensorflow as tf

import numpy.random as rnd

import numpy as np
# To plot pretty figures

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt
# to make this notebook's output stable across runs

def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)
reset_graph()



#CONSTRUCTION PHASE

n_steps = 21

n_inputs = 10 #same as rows

n_neurons = 200

n_outputs = 10 #same as rows

n_layers = 6



learning_rate = 0.001



X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])



#y has the same shape

y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])



#create the cells and the layers of the network

layers = [tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),output_size=n_outputs) for layer in range(n_layers)]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

outputs, states = tf.nn.dynamic_rnn(multi_layer_cell,X,dtype=tf.float32)



print(outputs.shape)

print(y.shape)



#define the cost function and optimization

loss = tf.reduce_mean(tf.square(outputs - y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)



#Adds an op to initialize all variables in the model

init = tf.global_variables_initializer()
#Import training data

import csv

from numpy import genfromtxt

import numpy as np

import tensorflow as tf

import pandas as pd

import random

from sklearn.preprocessing import Imputer



#read the csv into a dataframe

df=pd.read_csv('../input/train_1.csv', sep=',', error_bad_lines=False, warn_bad_lines=True)



#clean the data

df.drop(['Page'], axis=1, inplace = True)

df = df.replace(np.nan, 0, regex=True)



#grab the relevant rows

X_test = df.values.astype(int)

rows = print(len(X_test))



#number of rows to read 

rows = 10



#fetch the rows

X_test = df.values.astype(int)

X_test = X_test[:rows,0:]



#set the labelled data as the time step + 1

Y_test = X_test[:,1:]



#strip n numbers off the rows for reshape

X_test = X_test[:,:-4]

Y_test = Y_test[:,:-3]



#print("unshaped X data", X_test)

#print("unshaped X data shape", X_test.shape)



#print("unshaped Y data", Y_test)

#print("unshaped Y data shape", Y_test.shape)



for iteration in range(len(X_test)):

    plt.plot(X_test[iteration,:], label="page"+str(iteration+1))



plt.xlabel("Time")

plt.ylabel("Value")

plt.legend()

plt.show()    

    

Y_test = Y_test.reshape((-1, n_steps, n_inputs))

X_test = X_test.reshape((-1, n_steps, n_inputs))



#print("transformed X data", X_test[0])

#print("transformed X data", X_test[1])

#print("transformed Y data", Y_test[0])
#TRAINING PHASE



import numpy

n_epochs = 1000

batch_size = 10

instances = X_test.shape[0]

saver = tf.train.Saver()



print("RNN construction", "n_layers", n_layers, "n_neurons", n_neurons, "n_inputs", n_inputs, "n_outputs", n_outputs)

print("training dataset", "shape", X_test.shape, "instances", instances, "batch_size", batch_size)

print("training with", "n_epochs", n_epochs, "iterations (instances // batch_size)", (instances//batch_size))



#open a TensorFlow Session

with tf.Session() as sess:

    init.run()

    

    #Epoch is a single pass through the entire training set, followed by testing of the verification set.

    for epoch in range(n_epochs):

    

        #print("X_test",X_test)

        idxs = numpy.random.permutation(instances) #shuffled ordering

        #print(idxs)

        #print(idxs.shape)

        X_random = X_test[idxs]

        #print('X_random', X_random)

        Y_random = Y_test[idxs]

        #print('Y_random', Y_random)

    

        #Number of batches, here we exhaust the training set

        for iteration in range(instances // batch_size):   



            #get the next batch - we permute the examples in each batch

            #X_batch, y_batch = next_batch(batch_size, n_steps)

            X_batch = X_random[iteration * batch_size:(iteration+1) * batch_size]

            y_batch = Y_random[(iteration * batch_size):((iteration+1) * batch_size)]

            

            #print("iteration", iteration, "X_batch", X_batch)

            #print("iteration", iteration, "y_batch", y_batch)

            

            X_batch = X_batch.reshape((-1, n_steps, rows))

            y_batch = y_batch.reshape((-1, n_steps, rows))

        

            #feed in the batch

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        

            if (epoch % 100 == 0 and iteration == 0):

                #check contents of first example in each batch

                #print("iteration", iteration, "X_batch", X_batch[0])

                #print("iteration", iteration, "y_batch", y_batch[0])

                mse = loss.eval(feed_dict={X:X_batch, y:y_batch})

                print("epoch", epoch, "iteration", iteration, "\tMSE:", mse)

            

            #print(epoch)

    

    saver.save(sess, "./my_model") 
#Sequence generation using the model



max_iterations = 90 #number of time steps to look forward



with tf.Session() as sess:                        

    saver.restore(sess, "./my_model") 



    sequence = [[0] * n_steps] * rows

    for iteration in range(max_iterations):

        #print("iteration", iteration)

        

        X_batch = np.array(sequence[-n_steps*rows:]).reshape(-1,n_steps,rows)

        #print(X_batch)

        

        y_pred = sess.run(outputs, feed_dict={X: X_batch})

        #print("Y", y_pred)

        #print("numbers to be added", np.array(y_pred[0,n_steps-1]))

            

        sequence = np.append(sequence, y_pred[0,n_steps-1])      

        #print("sequence so far...", np.array(sequence))

        

    #print("end sequence", np.array(sequence))

    sequence[numpy.where(sequence<0)] = 0

    #print(sequence)

    sequence = np.array(sequence).reshape(-1,rows)

    #print(sequence)
plt.figure(figsize=(8,4))



#print(sequence[:,0])



for iteration in range(rows):

    plt.plot(sequence[:,iteration], label = "page"+str(iteration+1))



plt.xlabel("Time")

plt.ylabel("Value")

plt.legend()

plt.show()