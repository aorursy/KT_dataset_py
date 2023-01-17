import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
milk = pd.read_csv('/kaggle/input/monthy-milk/monthly-milk-production.csv',index_col='Month')
milk.head()
milk.index = pd.to_datetime(milk.index)
milk.describe()
milk.plot()
milk.info()
train_set = milk.head(156)
train_set.head()
test_set = milk.tail(12)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)
def next_batch(training_data,batch_size,steps):

    

    

    # Grab a random starting point for each batch

    rand_start = np.random.randint(0,len(training_data)-steps) 



    # Create Y data for time series in the batches

    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)



    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)
# y_batch[:<all rows>, :-1<from last>].reshape(-1, steps, 1), y_batch[:, 1:<from start>].reshape(-1, steps, 1) 

# we need 2 y_batch because of back propogating
# putting values to parameter

num_inputs = 1

num_time_steps = 12

num_neurons = 100

num_outputs = 1

learning_rate = 0.03

num_train_iterations = 4000

batch_size = 1
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])

y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])
cell = tf.contrib.rnn.OutputProjectionWrapper(

    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),

    output_size=num_outputs) 
# tf.nn.rnn_cell.LSTMCell above

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# above is deprecated use this one ===tf.keras.layers.RNN
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

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

    train_seed = list(train_scaled[-12:])

    

    ## Now create a for loop that 

    for iteration in range(12):

        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)

        y_pred = sess.run(outputs, feed_dict={X: X_batch})

        train_seed.append(y_pred[0, -1, 0])
# inverse transform data into normal form from scaling
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))
#new column on the test_set = "Generated" 
test_set['Generated'] = results
test_set.plot()
test_set