import numpy as np

import tensorflow as tf

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd
# def create_ts(start = '2001', n = 201, freq = 'M'):

#     rng = pd.date_range(start=start, periods=n, freq=freq)

#     ts = pd.Series(np.random.uniform(-18, 18, size=len(rng)), rng).cumsum()

#     return ts

# ts= create_ts(start = '2001', n = 402, freq = 'M')

# ts.tail(5)
ts = pd.read_csv("../input/YESBANK.NS.csv")[["Date", "Open"]]

ts = ts[len(ts)-202:]

print(len(ts))

ts.tail()
plt.plot(ts.index, ts["Open"])

plt.show()
series = np.array(ts["Open"])

n_windows = 20

n_input =  1

n_output = 1

len(series)
def create_batches(df, windows, input, output):

    ## Create X         

        print("DF",len(df))

        x_data = df[:size_train-output] # Select the data

        print("data", len(x_data))

        X_batches = x_data.reshape(-1, windows, input)  # Reshape the data 

    ## Create y

        y_data = df[output:size_train]

        y_batches = y_data.reshape(-1, windows, output)

        return X_batches, y_batches
size_train = 101

train = series[:size_train]

test = series[size_train:]

print(len(train), len(test))

X_train, y_train = create_batches(df = train,

                                      windows = n_windows,

                                      input = n_input,

                                      output = n_output)



print(len(train), X_train.shape, y_train.shape)



X_test, y_test = create_batches(df = test, windows = n_windows, input = n_input, output = n_output)

print(X_test.shape, y_test.shape)
plt.plot(np.ravel(y_train))

plt.show()

plt.plot(np.ravel(y_test))

plt.show()
tf.reset_default_graph()

r_neuron = 12



## 1. Construct the tensors

X = tf.placeholder(tf.float32, [None, n_windows, n_input])   

y = tf.placeholder(tf.float32, [None, n_windows, n_output])



## 2. create the model

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=r_neuron, activation=tf.nn.relu)   

rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)              



stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])          

stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)       

outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])   



## 3. Loss + optimization

learning_rate = 0.001  



loss = tf.reduce_sum(tf.square(outputs - y))    

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)         

training_op = optimizer.minimize(loss)                                          



init = tf.global_variables_initializer() 
def plot(y, y_, plt):

    plt.plot(pd.Series(np.ravel(y)), markersize=8, label="Actual", color='green')

    plt.plot(pd.Series(np.ravel(y_)), markersize=8, label="Forecast", color='red')




sess = tf.Session()

sess.run(init)
iteration = 1501

for iters in range(iteration):

    sess.run(training_op, feed_dict={X: X_train, y: y_train})

    if iters % 100 == 0:

        y_pred = sess.run(outputs, feed_dict={X: X_train})

        yt_pred = sess.run(outputs, feed_dict={X: X_test})

        fig, ax = plt.subplots(1,2, figsize = (14,4))

        plot(y_train, y_pred, ax[0])

        plot(y_test, yt_pred, ax[1])

        plt.show()



        mse = loss.eval(session=sess, feed_dict={X: X_train, y: y_train})

        print(iters, "\tMSE:", mse)



y_pred = sess.run(outputs, feed_dict={X: X_test})
X_test[-1], y_test[-1], y_pred[-1]