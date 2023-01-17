import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

data = pd.read_csv("../input/petr4.csv")
data = data.dropna()
data = data.iloc[:,1].values
plt.plot(data)
periods = 30 # days
# separate the records from stock prices excluding the latest 30 days (test data)
train_x = data[0:(len(data) - (len(data) % periods))]

train_x_batches = train_x.reshape(-1, periods, 1)  
# Uses 1 because it's only one independent feature
# created 41 batches with 30 records with 1 feature

# Using stock price for the "next day" of each 30 days batch as the dependent variable.
train_y = data[1:(len(data) - (len(data) % periods)) + 1] # increment 1 to get the "next day"
train_y_batches = train_y.reshape(-1, periods, 1)

print('train_x shape: ', train_x.shape)
print('train_y shape: ', train_y.shape)

print('train_x_batches shape: ', train_x_batches.shape)
print('train_y_batches shape: ', train_y_batches.shape)
test_x = data[-(periods + 1):]
test_x = test_x[:periods]
test_x = test_x.reshape(-1, periods, 1)
print('test_x shape: ', test_x.shape)

test_y = data[-(periods):]
test_y = test_y.reshape(-1, periods, 1)
print('test_y shape: ', test_y.shape)
import tensorflow as tf
tf.reset_default_graph() # memory clean
neurons_input = 1 # it's just one independent variable (feature)
neurons_hidden = 100
neurons_output = 1 # it's just on dependent variable

xph = tf.placeholder(tf.float32, [None, periods, neurons_input])
yph = tf.placeholder(tf.float32, [None, periods, neurons_output])

def create_cell():
    return tf.contrib.rnn.LSTMCell(num_units = neurons_hidden, activation = tf.nn.relu)

def create_cells():
    cells = tf.nn.rnn_cell.MultiRNNCell([create_cell() for i in range(4)]) # ten cells
    return tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob=0.1) # 10%

#cell = tf.contrib.rnn.BasicRNNCell(num_units = neurons_hidden, activation = tf.nn.relu)
cell = create_cells()
cell_output = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=1) # Dense Neural Network

rnn_output, _ = tf.nn.dynamic_rnn(cell_output, xph, dtype=tf.float32)

error = tf.losses.mean_squared_error(labels=yph, predictions=rnn_output)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(error)
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    
    for epoch in range(1000):
        _, cost = s.run([train, error], feed_dict = { xph: train_x_batches, yph: train_y_batches })
        if epoch % 100 == 0:
            print('Epoch: ', epoch + 1, ' - Cost error: ', cost)
            
    
    predictions = s.run(rnn_output, feed_dict = { xph: test_x })
    
import numpy as np
check_y = np.ravel(test_y) # reduction (1,30,1) to (30,)
check_predictions = np.ravel(predictions)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(check_y, check_predictions)
mae
# we got a high error of $5.5 into prices. 
# It's really bad, so we got a bad result using LSTM with Dropout. 
# So, it's a hard work to improve the algortihm changing the parameters, epochs, cells, etc.
plt.plot(check_y, '*', markersize=10, label = 'Real value')
plt.plot(check_predictions, 'o', markersize=10, label = 'Predictions')
plt.legend()

plt.plot(check_y, label = 'Real value')
plt.plot(check_predictions, label = 'Predictions')
plt.legend()