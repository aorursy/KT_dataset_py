import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import os

data = pd.read_csv("../input/export_dt.csv")
# reversing and indexing the data
data = data.iloc[::-1]
data.head()
data.dtypes
# fixing the date column
data.date = pd.to_datetime(data.date, errors='coerce')
# Cutting the annual only data
data = data[data.date >= '1971-01-01']
data = data.reset_index(drop=True)
# ploting just to take a look
data.plot()
data.head()
data = data.set_index('date')
# Setting up the training and testing sets
data.info()
train_set = data.head(-12)
test_set = data.tail(12)
train_set.tail()
test_set.head()
# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)
def next_batch(training_data, batch_size, steps, num_inputs, pred_len):
    
    ## Get a random starting place 
    rand_start = np.random.randint(0,len(training_data)-steps-pred_len)
    
    ## Get the training batch
    
    batchX = training_data[rand_start:(rand_start+steps), :num_inputs]
    batchX = batchX.reshape(1,steps,num_inputs)
    
    ## Get y batch
    batchY = training_data[rand_start+pred_len:(rand_start+steps+pred_len), :num_inputs]
    batchY = batchY.reshape(1,steps,num_inputs)
    
    return batchX , batchY
import tensorflow as tf
# Num of steps in each batch
num_time_steps = 12

# 100 neuron layer, play with this
num_neurons = 100

# learning rate you can play with this
learning_rate = 0.005 

# how many iterations to go through (training steps), you can play with this
num_train_iterations = 10000

# Size of the batch of data
batch_size = 1
def run_sess(forward, num_inputs):
    num_outputs = num_inputs
    
    # clear the graph
    
    tf.reset_default_graph()
    
    # Placeholders
    
    X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
    y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

    # create the layers
    
    cells = []
    for _ in range(3):
        cell = tf.nn.rnn_cell.GRUCell (num_units=num_neurons, activation = tf.nn.relu)  
    #         cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
        cells.append(cell)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

    # Create the output and states
    
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # Create the loss function and the trainer placeholders

    loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # Run the session
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
    
        for iteration in range(num_train_iterations):
            
            # getting the batches
        
            X_batch , y_batch = next_batch(train_scaled, batch_size, num_time_steps, num_inputs, pred_len = forward)
        
            sess.run(train, feed_dict={X: X_batch, y: y_batch})
            
            # keeping track of the progress of the function
        
            if iteration % 1000 == 0:
            
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                rmse = mse**0.5
                print(iteration, "\tRMSE:", rmse)
    
    
        # Create the different predictions
        
        # To deal with all the different shapes, split these up with an if statement. I believe that there is a more elegant way to do this. Also, the unlisting and then relisting is a basically duct taping the function together
        
        if num_inputs == 1:
            train_seed = list(train_scaled[-12:,0])
            
            for i in range(12):
                print(i)
                X_batch = np.array(train_seed[-num_time_steps:])
                X_batch = X_batch.reshape(1, num_time_steps, num_outputs)
                y_pred = sess.run(outputs, feed_dict={X: X_batch})
                y_pred = y_pred.reshape(num_time_steps)
                train_seed.append(y_pred[-1])
            
            # Seperating out the results on the basis of whether we are using a one step forward model or a 12 step forward model
            
            if forward > 1:
                return  np.stack((y_pred, np.zeros(12)), axis = 1)
            
            else:
                # attach a column of zeros so we can keep the shape consistent                
                return np.stack((np.array(train_seed[-12:]), np.zeros(12)), axis = 1)
#                 return train_seed[-12]
        
        else:
            train_seed = list(train_scaled[-12:, : ])
            for i in range(12):
                print(i)
                X_batch = np.array(train_seed[-num_time_steps:])
                X_batch = X_batch[:, 0:num_inputs]
                X_batch = X_batch.reshape(1, num_time_steps, num_outputs)
                y_pred = sess.run(outputs, feed_dict={X: X_batch})
                train_seed.append(y_pred[0, -1, :])
            
            # Seperating out the results on the basis of whether we are using a one step forward model or a 12 step forward model
            
            if forward > 1:
                return  y_pred.reshape(12,2)
            
            else:
                return np.array(train_seed[-12:])

# create a list of the values
results_list = [run_sess(forward = 1, num_inputs= 1), run_sess(forward = 12, num_inputs= 1), 
               run_sess(forward = 1, num_inputs= 2), run_sess(forward = 12, num_inputs= 2)]
results_unscaled = [scaler.inverse_transform(result) for result in results_list]
# putting the results in the data frame for easier graphing
test_set['Generated_1_1'] = results_unscaled[0][:,0]
test_set['Generated_12_1'] = results_unscaled[1][:,0]
test_set['Generated_1_2'] = results_unscaled[2][:,0]
test_set['Generated_12_2'] = results_unscaled[3][:,0]
test_set = test_set.drop('OIL', axis = 1)
test_set.head()
test_set.plot()
# Evaluate our models
from sklearn.metrics import mean_squared_error
rmse_1_1 = mean_squared_error(test_set.TEA_KOLKATA, test_set.Generated_1_1)**(0.5)
rmse_12_1 = mean_squared_error(test_set.TEA_KOLKATA, test_set.Generated_12_1)**(0.5)
rmse_1_2 = mean_squared_error(test_set.TEA_KOLKATA, test_set.Generated_1_2)**(0.5)
rmse_12_2 = mean_squared_error(test_set.TEA_KOLKATA, test_set.Generated_12_2)**(0.5)
# Print the values
print("RMSE of 1 period Foward, 1 input is {}".format(rmse_1_1))
print("RMSE of 12 periods Foward, 1 input is {}".format(rmse_12_1))
print("RMSE of 1 period Foward, 2 inputs is {}".format(rmse_1_2))
print("RMSE of 12 periods Foward, 2 inputs is {}".format(rmse_12_2))