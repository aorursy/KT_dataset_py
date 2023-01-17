import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
import datetime
import time
import_path = '../input/'
train_df = pd.read_csv(import_path + 'train.csv', skiprows=1, header=None, \
            names=['ip', 'app', 'device','os', 'channel', 'click_time', 'attributed_time', 'is_attributed'],\
            chunksize=1000000)
def process_chunk(sample_chunk):
    
    sample_chunk.index = pd.DatetimeIndex(sample_chunk['click_time'])
    sample_chunk['wday'] = sample_chunk.index.map(lambda x: x.weekday())
    sample_chunk['hour'] = sample_chunk.index.map(lambda x: x.hour)

    sample_chunk['(ip, wday, hour)'] = sample_chunk.apply(lambda row: (row['ip'], row['wday'], row['hour']), axis=1)
    sample_chunk['freq(ip, wday, hour)'] = sample_chunk.groupby(['(ip, wday, hour)'])['(ip, wday, hour)'].transform('count')
    sample_chunk = sample_chunk.drop(['(ip, wday, hour)'], axis=1)

    sample_chunk['(ip, hour, channel)'] = sample_chunk.apply(lambda row: (row['ip'], row['hour'], row['channel']), axis=1)
    sample_chunk['freq(ip, hour, channel)'] = sample_chunk.groupby(['(ip, hour, channel)'])['(ip, hour, channel)'].transform('count')
    sample_chunk = sample_chunk.drop(['(ip, hour, channel)'], axis=1)

    sample_chunk['(ip, hour, os)'] = sample_chunk.apply(lambda row: (row['ip'], row['hour'], row['os']), axis=1)
    sample_chunk['freq(ip, hour, os)'] = sample_chunk.groupby(['(ip, hour, os)'])['(ip, hour, os)'].transform('count')
    sample_chunk = sample_chunk.drop(['(ip, hour, os)'], axis=1)

    sample_chunk['(ip, hour, device)'] = sample_chunk.apply(lambda row: (row['ip'], row['hour'], row['device']), axis=1)
    sample_chunk['freq(ip, hour, device)'] = sample_chunk.groupby(['(ip, hour, device)'])['(ip, hour, device)'].transform('count')
    sample_chunk = sample_chunk.drop(['(ip, hour, device)'], axis=1)
    
    sample_chunk = sample_chunk.drop(['ip','click_time', 'attributed_time', 'wday'], axis=1)
    
    return sample_chunk
ops.reset_default_graph()
sess = tf.Session()
# Set RNN parameters
epochs = 25
batch_size = 100000
max_sequence_length = 9
rnn_size = 10
embedding_size = 50
learning_rate = 0.005
dropout_keep_prob = tf.placeholder(tf.float32)
# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
    

# Define Variable Functions (weights and bias)
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)
    

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)
    
    
# Create Placeholders
x_data = tf.placeholder(shape=[None, 9], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return(tf.nn.relu(layer))


#--------Create the first layer (50 hidden nodes)--------
weight_1 = init_weight(shape=[9, 50], st_dev=10.0)
bias_1   = init_bias(shape=[50], st_dev=10.0)
layer_1  = fully_connected(x_data, weight_1, bias_1)

#--------Create second layer (25 hidden nodes)--------
weight_2 = init_weight(shape=[50, 25], st_dev=10.0)
bias_2 = init_bias(shape=[25], st_dev=10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)


#--------Create third layer (5 hidden nodes)--------
weight_3 = init_weight(shape=[25, 10], st_dev=10.0)
bias_3 = init_bias(shape=[10], st_dev=10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)


#--------Create output layer (1 output value)--------
weight_4 = init_weight(shape=[10, 1], st_dev=10.0)
bias_4 = init_bias(shape=[1], st_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

# Declare loss function:
loss = tf.reduce_mean(tf.square(y_target - final_output))

# Declare optimizer
my_opt = tf.train.AdamOptimizer(0.005)
train_step = my_opt.minimize(loss)
%matplotlib inline

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []

for idx_chunk, chunk in enumerate(train_df):
    print('chunk n°{} is being processing...'.format(idx_chunk))
    chunk = process_chunk(chunk)
    print('chunk processed.')
    
    x_vals = np.array(chunk.drop(['is_attributed'], axis=1))
    y_vals = np.array(chunk['is_attributed'])
    
    shuffled_ix = np.random.permutation(np.arange(len(x_vals)))
    x_shuffled = x_vals[shuffled_ix]
    y_shuffled = y_vals[shuffled_ix]
    
    # Split train/test set
    ix_cutoff = int(len(y_vals)*0.80)
    x_vals_train, x_vals_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
    y_vals_train, y_vals_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
    
    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

    for i in range(epochs):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_loss.append(test_temp_loss)
        if (i+1) % 10 == 0:
            print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

    if idx_chunk % 10 == 0:
        # Plot loss (MSE) over time
        plt.plot(loss_vec, 'k-', label='Train Loss')
        plt.plot(test_loss, 'r--', label='Test Loss')
        plt.semilogy()
        plt.title('Loss (MSE) per Generation')
        plt.legend(loc='upper right')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.show()
    
    if idx_chunk == 50: # Equivalent to 50 Mo lines of processed rows
        break