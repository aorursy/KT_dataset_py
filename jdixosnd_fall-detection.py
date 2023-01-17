# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from scipy.misc import imread

from sklearn.metrics import accuracy_score

import tensorflow as tf

import time



print(os.listdir("../input"))

dataset = pd.read_csv("../input/falldeteciton.csv")

Y = dataset['ACTIVITY']

ds = dataset.drop('ACTIVITY',1)

ds = ds.drop('TIME',1)

X = (ds - pd.DataFrame.mean(ds)) / pd.DataFrame.std(ds)

Y = pd.get_dummies(Y,sparse=True)

# To stop potential randomness

seed = 128

rng = np.random.RandomState(seed)

X_train,X_test,X_cv = X[:13105],X[13105:14743],X[14743:15382]

Y_train,Y_test,Y_cv = Y[:13105],Y[13105:14743],Y[14743:15382]

# Any results you write to the current directory are saved as output.





### set all variables



# number of neurons in each layer

input_num_units = 5

hidden_num_units_1 = 25

hidden_num_units_2 = 55

hidden_num_units_3 = 55

hidden_num_units_4 = 60

hidden_num_units_5 = 60

#hidden_num_units_6 = 60

output_num_units = 6



# define placeholders

x = tf.placeholder(tf.float32, [None, input_num_units])

y = tf.placeholder(tf.float32, [None, output_num_units])



# set remaining variables

epochs = 80

batch_size = 1

learning_rate = 0.008

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)



weights = {

    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units_1], seed=seed)),

    'hidden2': tf.Variable(tf.random_normal([hidden_num_units_1, hidden_num_units_2], seed=seed)),

    'hidden3': tf.Variable(tf.random_normal([hidden_num_units_2, hidden_num_units_3], seed=seed)),

    'hidden4': tf.Variable(tf.random_normal([hidden_num_units_3, hidden_num_units_4], seed=seed)),

    'hidden5': tf.Variable(tf.random_normal([hidden_num_units_4, hidden_num_units_5], seed=seed)),

    

   # 'hidden6': tf.Variable(tf.random_normal([hidden_num_units_5, hidden_num_units_6], seed=seed)),

    'output': tf.Variable(tf.random_normal([hidden_num_units_5, output_num_units], seed=seed))

}



biases = {

    'hidden1': tf.Variable(tf.random_normal([hidden_num_units_1], seed=seed)),

    'hidden2': tf.Variable(tf.random_normal([hidden_num_units_2], seed=seed)),

    'hidden3': tf.Variable(tf.random_normal([hidden_num_units_3], seed=seed)),

    'hidden4': tf.Variable(tf.random_normal([hidden_num_units_4], seed=seed)),

    'hidden5': tf.Variable(tf.random_normal([hidden_num_units_5], seed=seed)),

  #  'hidden6': tf.Variable(tf.random_normal([hidden_num_units_6], seed=seed)),

    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))

}



hidden_layer_1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])

hidden_layer_1 = tf.nn.relu(hidden_layer_1)

hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['hidden2']), biases['hidden2'])

hidden_layer_2 = tf.nn.relu(hidden_layer_2)

hidden_layer_3 = tf.add(tf.matmul(hidden_layer_2, weights['hidden3']), biases['hidden3'])

hidden_layer_3 = tf.nn.relu(hidden_layer_3)

hidden_layer_4 = tf.add(tf.matmul(hidden_layer_3, weights['hidden4']), biases['hidden4'])

hidden_layer_4 = tf.nn.relu(hidden_layer_4)

hidden_layer_5 = tf.add(tf.matmul(hidden_layer_4, weights['hidden5']), biases['hidden5'])

hidden_layer_5 = tf.nn.relu(hidden_layer_5)

#hidden_layer_6 = tf.add(tf.matmul(hidden_layer_5, weights['hidden6']), biases['hidden6'])

#hidden_layer_6 = tf.nn.relu(hidden_layer_6)



output_layer = tf.matmul(hidden_layer_5, weights['output']) + biases['output']



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer,labels= y))



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



tf.trainable_variables()



init = tf.initialize_all_variables()





def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

    assert inputs.shape[0] == targets.shape[0]

    if shuffle:

        indices = np.arange(inputs.shape[0])

        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):

        if shuffle:

            excerpt = indices[start_idx:start_idx + batchsize]

        else:

            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]



start = time.time()

costs = []

iterations = []

with tf.Session() as sess:

    # create initialized variables

    sess.run(init)

    

    ### for each epoch, do:

    ###   for each batch, do:

    ###     create pre-processed batch

    ###     run optimizer by feeding batch

    ###     find cost and reiterate to minimize

    

    for epoch in range(epochs):

        avg_cost = 0

        total_batch = int(X_train.shape[0]/batch_size)

        for batch in iterate_minibatches(X_train,Y_train,100,shuffle=False):

            #print(batch)

            x_batch, y_batch = batch

            _, c = sess.run([optimizer, cost], feed_dict = {x: x_batch, y: y_batch})

            avg_cost += c / total_batch

        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))

        costs.append("{:.2f}".format(avg_cost))

        iterations.append(epoch+1)

    print ("\nTraining complete!")

    

    

    # find predictions on val set

    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

    print ("Train Validation Accuracy:", accuracy.eval({x: X_train.values.reshape(-1, input_num_units), y: pd.get_dummies(Y_train,sparse=True)}))

    print ("CV Validation Accuracy:", accuracy.eval({x: X_cv.values.reshape(-1, input_num_units), y: pd.get_dummies(Y_cv,sparse=True)}))

    

    done = time.time()

    elapsed = done - start

    print("Time taken: ",elapsed)

    predict = tf.argmax(output_layer, 1)

    pred = predict.eval({x: X_cv.values.reshape(-1, input_num_units)})

    

    import matplotlib.pyplot as plt

    costs

    iterations

    plt.plot(iterations,costs)

    

    