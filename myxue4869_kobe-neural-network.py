# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## import packages and modules
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import math
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import sklearn
from sklearn.model_selection import train_test_split
import time
# show plots inline
%matplotlib inline
## dataset path
filename = "../input/data.csv"
## set default figure size
figure_size = (15,10)
# set max display row number
pd.set_option('max_rows',5)

## load dataset
# set column 'shot_id' as index because it is subjective and unique
df = pd.read_csv(filename, parse_dates=['game_date'], index_col='shot_id')
# view first 3 lines
print(df.head(3))
# list all features
print(df.columns.values)
# response variable
response_variable = 'shot_made_flag'
## columns not needed
notNeeded = []
## dummy variables
dummy_var = []
## craete new features and delete unnecessary ones
# action_type
for elem in df['action_type'].unique():
    df[str(elem)] = (df['action_type'] == elem).astype(int)
    dummy_var.append(str(elem))
notNeeded.append('action_type')
# combined shot type 
for elem in df['combined_shot_type'].unique():
    df[str(elem)] = (df['combined_shot_type'] == elem).astype(int)
    dummy_var.append(str(elem))
notNeeded.append('combined_shot_type')
# game event and game IDs not needed, subjective index
notNeeded.extend(['game_event_id','game_id'])
# lat, lon, loc_x, loc_y
sns.set_style('whitegrid')
sns.pairplot(df, vars=['loc_x', 'lon'], hue='shot_made_flag',size = 3)
sns.pairplot(df, vars=[ 'loc_y', 'lat'], hue='shot_made_flag',size = 3)
sns.set_style('whitegrid')
sns.pairplot(df, vars=['loc_x', 'loc_y'], hue='shot_made_flag')
#loc_x and lon are correlated, also loc_y and lat, so we'll drop lon and lat.
notNeeded.extend(['lon','lat'])

# minutes_remaining and seconds_remaining can be put in one column named time_remaining.
df['timeRemaining'] = 60 * df.loc[:,'minutes_remaining'] + df.loc[:,'seconds_remaining']
notNeeded.extend(['minutes_remaining','seconds_remaining'])
# season, just keep the year when season started
df['season'] = df['season'].apply(lambda x: x[:4])
# convert column to integer
df['season'] = pd.to_numeric(df['season'])
# shot distance, seems like shot_distance is just floored distance calculated from x- and y- location of a shot,
# so we'll use more precise measure and drop floored one.
distance = pd.DataFrame({'trueDistance': np.sqrt((df['loc_x']/10)** 2 + (df['loc_y']/10) ** 2),
                       'shotDistance': df['shot_distance']})
print(distance.head(5))
df['shotDistance'] = distance['trueDistance']
notNeeded.append('shot_distance')
# shot type
df['3ptGoal'] = df['shot_type'].str.contains('3PT').astype('int')
dummy_var.append('3ptGoal')
notNeeded.append('shot_type')
#shot_zone_range is just putting shot_distance in 5 bins, not needed
notNeeded.append('shot_zone_range')
# shot zone area and basic
for elem in df['shot_zone_area'].unique():
    df[str(elem)] = (df['shot_zone_area'] == elem).astype(int)
    dummy_var.append(str(elem))
notNeeded.append('shot_zone_area')
for elem in df['shot_zone_basic'].unique():
    df[str(elem)] = (df['shot_zone_basic'] == elem).astype(int)
    dummy_var.append(str(elem))
notNeeded.append('shot_zone_basic')
# team id and team name, consistent within the dataset
notNeeded.extend(['team_id','team_name'])
# game date
# convert game_date to datetime format, and then split it to year, month and weekday (0 = Monday, 6 = Sunday)
df['game_date'] = pd.to_datetime(df['game_date'])
df['game_year'] = df['game_date'].dt.year
df['game_month'] = df['game_date'].dt.month
df['game_day'] = df['game_date'].dt.dayofweek
# create indicate variables for month and weekday
for elem in df['game_month'].unique():
    pass
notNeeded.append('game_date')
# matchup and opponent
# matchup and opponent columns give as almost the same data - matchup tells us if the game was home or away (depending if it is '@' or 'vs'), 
# so we'll make a new column with that info and then we can drop matchup column.
df['homeGame'] = df['matchup'].str.contains('vs').astype(int)
notNeeded.append('matchup')
for elem in df['opponent'].unique():
    df[str(elem)] = (df['opponent'] == elem).astype(int)
    dummy_var.append(str(elem))
notNeeded.append('opponent')
# finally drop all not needed columns:
df = df.drop(notNeeded,axis=1)
## split into training set and predict set
train_df = df.loc[df['shot_made_flag'].notnull()]
predict_df = df.loc[df['shot_made_flag'].isnull()]
# variables in dummy_var is a sparse matrix
print(train_df.head(3))
print(predict_df.head(3))
# normalizing for not dummy variables
Y = df['shot_made_flag'].as_matrix()
Y = Y.reshape(Y.shape[0],1)   
X = df.drop(['shot_made_flag'], axis=1)
max_x = []
for c in X.columns.values:
    if c not in dummy_var:
        max_x.append(df[str(c)].max())
    else:
        max_x.append(1)
## labels Y and X
# training set
train_Y = train_df['shot_made_flag'].as_matrix()
train_Y = train_Y.reshape(train_Y.shape[0],1)      
train_X = train_df.drop(['shot_made_flag'], axis=1)
train_X = train_X/max_x
# test set
predict_Y = predict_df['shot_made_flag'].as_matrix()
predict_Y = predict_Y.reshape(predict_Y.shape[0],1)
predict_X = predict_df.drop(['shot_made_flag'], axis=1)
predict_X = predict_X/max_x
## display training and test dataframe
print(train_X.head(3))
print(predict_X.head(3))
train_X = train_X.as_matrix()
predict_X = predict_X.as_matrix()
## transpose matrix
train_X = train_X.T
train_Y = train_Y.T
predict_X = predict_X.T
predict_Y = predict_Y.T
## print info
print ("number of training examples = " + str(train_X.shape[1]))
print ("number of test examples = " + str(predict_X.shape[1]))
print ("train_X shape: " + str(train_X.shape))
print ("train_Y shape: " + str(train_Y.shape))
print ("predict_X shape: " + str(predict_X.shape))
print ("predict_Y shape: " + str(predict_Y.shape))
## store processed and normalized dataframe
df = X/max_x
df['shot_made_flag'] = Y
df['shot_id'] = df.index.values
print(df.head(3))
output_file = 'processed.csv'
df.to_csv(output_file, index = False)
## define neural network function
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.    
    Arguments:
    n_x -- scalar, size of an image vector
    n_y -- scalar, number of classes   
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(tf.float32, shape = [n_x,None])
    Y = tf.placeholder(tf.float32, shape = [n_y,None])    
    return X, Y
## define neural network function
def initialize_parameters(nn):
    """
    layer = len(nn)-1
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n1, n0]
                        b1 : [n1, 1]
                        W2 : [n2, n1]
                        b2 : [n2, 1]
                        ...
                        W_layer : [n(layer), n(layer-1)]
                        b_layer : [n(layer), 1]    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, ...
    """
    parameters = {}
    for i in range(len(nn)-1):
        parameters['W' + str(i+1)] = tf.get_variable('W'+str(i+1), [nn[i+1], nn[i]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b' + str(i+1)] = tf.get_variable('b' + str(i+1), [nn[i+1],1], initializer = tf.zeros_initializer())
    return parameters
## define neural network function
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", ...
                  the shapes are given in initialize_parameters
    Returns: 
    Z(len(parameters)/2) -- the output of the last LINEAR unit
    """
    length = int(len(parameters)/2)
    Z = tf.add(tf.matmul(parameters['W1'], X),parameters['b1'])
    A = tf.nn.relu(Z)  
    for i in range(2, length):
        Z = tf.add(tf.matmul(parameters['W'+str(i)], A),parameters['b'+str(i)])
        A = tf.nn.relu(Z)  
    Z = tf.add(tf.matmul(parameters['W'+str(length)], A),parameters['b'+str(length)])
    return Z
## deifne neural network function
def compute_cost(Z_end, Y):
    """
    Computes the cost  
    Arguments:
    Z_end -- output of forward propagation (output of the last LINEAR unit), of shape (1, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z_end
    Returns:
    cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z_end)
    labels = tf.transpose(Y)   
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))   
    return cost
## define neural network function
def model(X_train, Y_train, X_test, nn, learning_rate = 0.001,
          num_epochs = 1500, print_cost = True):
    """
    Implements a len(nn)-1 layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->...->LINEAR->SIGMOID. 
    Arguments:
    X_train -- training set
    Y_train -- test set
    nn -- input layer + number of neurals in each layer
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 100 epochs
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x,n_y)
    # Initialize parameters
    parameters = initialize_parameters(nn)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z_end = forward_propagation(X, parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z_end, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    # Initialize all the variables
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:  
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            # decrease learning rate every 1000 iterations to avoid oscillation
            if epoch%1000 == 1:
                learning_rate_now = learning_rate * np.exp(-epoch/num_epochs)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_now).minimize(cost)
            _ , tmp_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            epoch_cost = tmp_cost
            # Print the cost every 100 epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 10 == 0:
                costs.append(epoch_cost)     
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.show()
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        return parameters
## training
# train
start_time = time.time()
neural_num = [train_X.shape[0], 25, 12, 1]
parameters = model(train_X, train_Y, predict_X, neural_num, num_epochs = 10000, learning_rate = 0.1)
end_time = time.time()
print("Neural network training time consumed: %lf secs" % (end_time - start_time))
## accuracy and prediction
z_end = forward_propagation(tf.cast(predict_X, tf.float32), parameters)
predict_y = tf.sigmoid(z_end)
with tf.Session() as sess:
    predict_y = sess.run(predict_y)
    print(predict_y)

## submit
mask = df['shot_made_flag'].isnull()
submission = pd.DataFrame({"shot_id":df[mask].index, "shot_made_flag":predict_y[0]})
submission.sort_values('shot_id',  inplace=True)
submission.to_csv("submission.csv",index=False)
## nect step:
# further feature engineering 
# implement neural network with keras
# cross-validation
# ensemble
# xgboost

