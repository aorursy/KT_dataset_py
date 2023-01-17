import tensorflow as tf

import numpy.random as rnd
#CONSTRUCTION PHASE

n_steps = 28

n_inputs = 28 #28 features, 1 per pixel

n_neurons = 150

n_outputs = 10

n_layers = 3



learning_rate = 0.001



#create input and output placeholders

#X has shape [    t = 0              t = 1          t = 28 (n_steps)

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  instance 1

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  instance 2

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  instance 3

#             ...

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  total instances in batch (None | Undefined)

#            ]

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])



#Y is an integer 0 - 9

y = tf.placeholder(tf.int32, [None])



#create the cells and the layers of the network

layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)



#For classification, we only want the output activation at the last time step so we use 'outputs' 

#We transpose so that the time axis is first and use tf.gather() for selecting the last index, x28. 



#X has shape [    instance 1       instance 2      instance n (150)

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  t = 0

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  t = 1

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  t = 2

#             ...

#             [[x1,x2,...,x28],[x1,x2,...,x28],...,[x1,x2,...,x28]]  t = 28

#            ]

outputs = tf.transpose(outputs, [1, 0, 2])



#gather the final values across t = 28

last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)



#Create Softmax layer with XEntropy 

logits = tf.layers.dense(last, n_outputs)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)



#Compute the loss, mean xentropy score

loss = tf.reduce_mean(xentropy)



#Set the optimizer

#Here we use Adaptive Momement Estimation which keeps track of an exponentially decaying average of past gradients

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)



#Compute accuracy

#Does the highest logit correspond to the target class?

#in_top_k returns a 1D tensor full of boolean values

correct = tf.nn.in_top_k(logits, y, 1)

#We cast to floating point numbers and then take the mean to compute an accuracy score

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



#Adds an op to initialize all variables in the model

init = tf.global_variables_initializer()
#Import training data

import csv

from numpy import genfromtxt

import tensorflow as tf

import pandas as pd

df=pd.read_csv('../input/train.csv', sep=',')



y_test = df['label'].values

#print(y_test)



df.drop(['label'], axis=1, inplace = True)

X_test = df.values

X_test = X_test.reshape((-1, n_steps, n_inputs))



#print(X_test)

#print(X_test.shape)

print("training data imported")
import numpy



#TRAINING PHASE

n_epochs = 50

batch_size = 150

examples = 42000



saver = tf.train.Saver()



#open a TensorFlow Session

with tf.Session() as sess:

    init.run()

    

    #Epoch is a single pass through the entire training set, followed by testing of the verification set.

    for epoch in range(n_epochs):

    

        idxs = numpy.random.permutation(examples) #shuffled ordering

        #print(idxs)

        #print(idxs.shape)

        X_random = X_test[idxs]

        #print('X_random', X_random)

        Y_random = y_test[idxs]

        #print('Y_random', Y_random)

    

        #Number of batches, here we exhaust the training set

        for iteration in range(examples // batch_size):   



            #get the next batch - we permute the 

            X_batch = X_random[iteration * batch_size:(iteration+1) * batch_size]

            

            #print(X_batch[1,:])

            #print(iteration * batch_size,(iteration+1) * batch_size)

            

            y_batch = Y_random[iteration * batch_size:(iteration+1) * batch_size]

                        

            #obtain batch of specified batch size

            #X_batch, y_batch = X_test.next_batch(batch_size) 

            

            #reshape the array to size n_steps * n_inputs

            X_batch = X_batch.reshape((-1, n_steps, n_inputs)) 

            

            #feed in the batch

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        

        #compute accuracy of RNN against Training Set and Test Set

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    

    saver.save(sess, "./my_model") 
#Import Test data

import csv

from numpy import genfromtxt

import tensorflow as tf

import pandas as pd

#df=pd.read_csv('test.csv', sep=',', usecols=range(0, 784), error_bad_lines=True, warn_bad_lines=True)

df=pd.read_csv('../input/test.csv', sep=',', warn_bad_lines=True)

print(df.shape)

#y_test = df['label'].values

#print(y_test)



#df.drop(['label'], axis=1, inplace = True)

X_test = df.values

X_test = X_test.reshape((-1, n_steps, n_inputs))



print("test data imported")

#print(X_test)

#print(X_test.shape)
import numpy as np

import pandas as pd



#PREDICTION PHASE

y = tf.placeholder(tf.int32, [None])

data = []

                

#open a TensorFlow Session

with tf.Session() as sess:

    saver.restore(sess, "./my_model")

    

    #print(X_test[1,:])

    

    length = X_test.shape[0]

    print("examples found:", length)

    

    for num in range(length):

    

        X_predict = X_test[num,:].reshape((-1, n_steps, n_inputs))

            

        #feed in the example

        classification = sess.run(logits, feed_dict={X: X_predict}).argmax()

        #print(classification)

        data.append([num+1, classification])



    print(data)
#write the output to file

import pandas as df



columns = ['ImageId', 'Label']

df = pd.DataFrame(data, columns=columns)

df.to_csv('out.csv',columns=columns, index=False)