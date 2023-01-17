#libraries

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
#input data

observations = 1000



xs = np.random.uniform(low=-10, high=10, size=(observations, 1))

zs = np.random.uniform(-10, 10, (observations, 1))

generated_inputs = np.column_stack((xs, zs))



noise = np.random.uniform(-1, 1, (observations, 1))

generated_targets = 2*xs - 3*zs + 5 + noise



#np.savez(file name, label1=array1, label2=array2...)

#method for saving multidimensional arrays in .npz format

#data -> preprocess -> save in .npz

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)
#solving with tf

input_size = 2

output_size = 1



#tf.paceholder(dataType, [dimensions]) :

#It is where we feed the data. Data in our dataset go into placeolder

inputs = tf.placeholder(tf.float32, [None, input_size]) #'NONE' MEANS 'WE ARE NOT SPECIFING DIMENSEION' TF will keep track of number of obsvns

targets = tf.placeholder(tf.float32, [None, output_size]) #'NONE' MEANS 'WE ARE NOT SPECIFING DIMENSEION'



#tf.variable :

#Throughout the iterations, each 'variable' is always used for each iteration.

#Whereas, different 'placeholders'are used with each iter

weights = tf.Variable(tf.random.uniform([input_size, output_size], minval=-10, maxval=10))

biases = tf.Variable(tf.random.uniform([output_size], minval=-0.1, maxval=0.1))



#tf.matmul

outputs = tf.matmul(inputs,weights) + biases



#NOTE : NOTHING WILL HAPPEN IF WE RUN THIS BLOCK OF CODE. NO CALCULATIONS, NO INITIALISATIONS.

#       ONLY THE LOGIC IS LAYED WITH WHICH OUR ALGORITHM WILLWORK. THAT'S WHAT TF LOGIC IS ALL ABOUT
#OBJECTIVE FUNC and OPTIMISATION ALGORITHM

mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs) / 2.

optimize = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)
#NOTE : Nothing was don by TF. Till now, we were laying logic. From here, It will start xecuting 



#PREPARE FOR XECUTION

#tf.InteractiveSession():

#used when we want to xecute something. Training happens in sessions.

sess = tf.InteractiveSession()



#INITIALISATION : ASSIGNING INITIAL VALUES FOR VARIABLES

#Tf.global_variables_initializer() :

#Initilizes all tensor objects marked as 'variables'

initializer = tf.global_variables_initializer()



#Did we initialize? No! We didn't execute. 

#sess.run(what we want to run) : 

#method for executing

sess.run(initializer)



#LOADING DATA

#np.load(file) :

#This will only load data from npz, npy or pickled files inta an object

training_data = np.load('TF_intro.npz')





#LEARNING



#EPOCH : Each itertion over full dataset



#feed_dict={placeholder1:data, placeholder2:data} :

#Tells algorith how data is going to be fed



# _ : Underscore is a special symbol to disregard return value of a function or methog

#for example, optimizer does not return a value. It always returns 'None'



#NOTE : The code is so generic that we can apply this to any ML Algorithm

for e in range(100):

    _, curr_loss = sess.run([optimize, mean_loss],

                           feed_dict={inputs:training_data['inputs'], targets:training_data['targets']})

    print(curr_loss)

    
#PLOTTING

out = sess.run([outputs],

              feed_dict={inputs: training_data['inputs']})

plt.plot(np.squeeze(out), np.squeeze(training_data['targets']))

plt.xlabel('outputs')

plt.ylabel('targets')

plt.show()
