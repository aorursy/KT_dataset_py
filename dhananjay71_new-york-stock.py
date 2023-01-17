# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import math

import sklearn

import sklearn.preprocessing

import datetime

import os

import matplotlib.pyplot as plt

import tensorflow as tf



valid_set_size_percentage = 10 

test_set_size_percentage = 10 









# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)

df.info()

df.head()
df.tail()

def normalize_data(df):

    min_max_scaler = sklearn.preprocessing.MinMaxScaler()

    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))

    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))

    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))

    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))

    return df



def load_data(stock, seq_len):

    data_raw = stock.as_matrix() # convert to numpy array

    data = []

    

    for index in range(len(data_raw) - seq_len): 

        data.append(data_raw[index: index + seq_len])

    

    data = np.array(data);

    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  

    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));

    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    

    x_train = data[:train_set_size,:-1,:]

    y_train = data[:train_set_size,-1,:]

    

    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]

    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]

    

    x_test = data[train_set_size+valid_set_size:,:-1,:]

    y_test = data[train_set_size+valid_set_size:,-1,:]

    

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]



df_stock = df[df.symbol == 'EQIX'].copy()

df_stock.drop(['symbol'],1,inplace=True)

df_stock.drop(['volume'],1,inplace=True)



cols = list(df_stock.columns.values)

print('df_stock.columns.values = ', cols)



df_stock_norm = df_stock.copy()

df_stock_norm = normalize_data(df_stock_norm)



seq_len = 20 

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)

print('x_train.shape = ',x_train.shape)

print('y_train.shape = ', y_train.shape)

print('x_valid.shape = ',x_valid.shape)

print('y_valid.shape = ', y_valid.shape)

print('x_test.shape = ', x_test.shape)

print('y_test.shape = ',y_test.shape)



index_in_epoch = 0;

perm_array  = np.arange(x_train.shape[0])

np.random.shuffle(perm_array)



def get_next_batch(batch_size):

    global index_in_epoch, x_train, perm_array   

    start = index_in_epoch

    index_in_epoch += batch_size

    

    if index_in_epoch > x_train.shape[0]:

        np.random.shuffle(perm_array) 

        start = 0 

        index_in_epoch = batch_size

        

    end = index_in_epoch

    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]



# parameters

n_steps = seq_len-1 

n_inputs = 4 

n_neurons = 200 

n_outputs = 4

n_layers = 2

learning_rate = 0.001

batch_size = 50

n_epochs = 100 

train_set_size = x_train.shape[0]

test_set_size = x_test.shape[0]



tf.reset_default_graph()



X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

y = tf.placeholder(tf.float32, [None, n_outputs])



layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)

          for layer in range(n_layers)]



                                                                     

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)



stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 

stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)

outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

outputs = outputs[:,n_steps-1,:] 

                                              

loss = tf.reduce_mean(tf.square(outputs - y)) 

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 

training_op = optimizer.minimize(loss)

                                              

# run graph

with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    for iteration in range(int(n_epochs*train_set_size/batch_size)):

        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 

        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 

        if iteration % int(5*train_set_size/batch_size) == 0:

            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 

            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 

            print('%.2f epochs: MSE train/valid = %.6f/%.6f'%(

                iteration*batch_size/train_set_size, mse_train, mse_valid))



    y_train_pred = sess.run(outputs, feed_dict={X: x_train})

    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})

    y_test_pred = sess.run(outputs, feed_dict={X: x_test})

print(y_test_pred)

plt.plot(y_test, color = 'red', label = 'Actual stock value')

plt.plot(y_test_pred, color = 'blue', label = 'Predicted stock value')

plt.title('Stock prediction')

plt.xlabel('Time')

plt.ylabel('Stock price')

plt.legend()

plt.show()