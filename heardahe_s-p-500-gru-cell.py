# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from math import pi

from matplotlib import pyplot as plt



from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/all_stocks_5yr.csv",index_col='date')

df.info()
#df.date = pd.to_datetime(df.date)

#df['month'] = df['date'].dt.month

#df['year'] = df['date'].dt.year
df.head(10)
df.tail(10)
df.isnull().sum()

df[df.open.isnull()]
df = df.dropna()

df.info()
#df = df.set_index('date')

#df = pd.to_datetime(df.index)

train_set = df[(df.index > '2016-12-31') & (df.index < '2018-01-01') & (df.Name == 'MSFT')]

test_set  = df[(df.index > '2017-12-31') & (df.index < '2018-01-31') & (df.Name == 'MSFT')]





train_set = train_set.filter(['close'])

test_set = test_set.filter(['close'])





scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_set)

test_scaled = scaler.transform(test_set)
def next_batch(training_data,batch_size,steps):

    

    

    # Grab a random starting point for each batch

    rand_start = np.random.randint(0,len(training_data)-steps) 



    # Create Y data for time series in the batches

    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)



    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)





# Just one feature, the time series

num_inputs = 1

# Num of steps in each batch

num_time_steps = 12

# 100 neuron layer, play with this

num_neurons = 300

# Just one output, predicted time series

num_outputs = 1



## You can also try increasing iterations, but decreasing learning rate

# learning rate you can play with this

learning_rate = 0.0001

# how many iterations to go through (training steps), you can play with this

num_train_iterations = 8000

# Size of the batch of data

batch_size = 1





X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])

y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])



cell = tf.contrib.rnn.OutputProjectionWrapper(

    tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu),

    output_size=num_outputs) 



outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)



loss = tf.reduce_mean(tf.square(outputs - y)) # MSE

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)



init = tf.global_variables_initializer()



saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto()) as sess:

    sess.run(init)

    

    for iteration in range(num_train_iterations):

        

        X_batch, y_batch = next_batch(train_scaled,batch_size,num_time_steps)

        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        

        if iteration % 100 == 0:

            

            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})

            print(iteration, "\tMSE:", mse)

    

    # Save Model for Later

    saver.save(sess, "./ex_time_series_model")
with tf.Session() as sess:

    

    # Use your Saver instance to restore your saved rnn time series model

    saver.restore(sess, "./ex_time_series_model")



    # Create a numpy array for your genreative seed from the last 12 months of the 

    # training set data. Hint: Just use tail(12) and then pass it to an np.array

    train_seed = list(train_scaled[-20:])

    

    ## Now create a for loop that 

    for iteration in range(20):

        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)

        y_pred = sess.run(outputs, feed_dict={X: X_batch})

        train_seed.append(y_pred[0, -1, 0])

        

        

        

results = scaler.inverse_transform(np.array(train_seed[20:]).reshape(20,1))

test_set['Generated'] = results

test_set



test_set.plot()