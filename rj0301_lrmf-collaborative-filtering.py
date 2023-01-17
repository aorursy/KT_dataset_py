import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.framework import ops
import math
tf.compat.v1.disable_eager_execution()
ratings_df = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
n_ratings, _ = ratings_df.shape
print(ratings_df.head())
print(ratings_df.tail())
print(n_ratings)
train_size = int(0.8*n_ratings)
shuffled_indexes = shuffle(range(n_ratings))
df = ratings_df.iloc[shuffled_indexes[:train_size]]
test_df = ratings_df.iloc[shuffled_indexes[train_size:]]
print(df.shape)
print(test_df.shape)
df_flag = df.pivot(index='movieId', columns='userId', values='rating').isna()==False
df_flag = df_flag.values
col_names = df.pivot(index='movieId', columns='userId', values='rating').columns
row_names = df.pivot(index='movieId', columns='userId', values='rating').index
y_matrix = df.pivot(index='movieId', columns='userId', values='rating').fillna(0).values
n_movies, n_users = y_matrix.shape
print(n_movies, n_users)
def create_placeholders(n_y):

    ### START CODE HERE ### (approx. 2 lines)
    Y = tf.compat.v1.placeholder(tf.float32,shape=(n_y, None), name='Y')    
    Y_flag = tf.compat.v1.placeholder(tf.float32, shape=(n_y, None), name="Y_flag")
    ### END CODE HERE ###
    
    return Y, Y_flag
Y, Y_flag = create_placeholders(12288)
print ("Y = " + str(Y))
print ("Y_flag = " + str(Y_flag))
def initialize_parameters(item, user):

    tf.compat.v1.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    X_mat = tf.compat.v1.get_variable("X_mat", [item, 100], initializer = tf.initializers.GlorotUniform(seed = 1))
    theta_mat = tf.compat.v1.get_variable("theta_mat", [100,user], initializer = tf.initializers.GlorotUniform(seed = 1))
    ### END CODE HERE ###

    parameters = {"X_mat": X_mat,
                  "theta_mat": theta_mat}
    
    return parameters
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    parameters = initialize_parameters(500, 600)
    print("X_mat = " + str(parameters["X_mat"]))
    print("theta_mat = " + str(parameters["theta_mat"]))
def compute_cost(X, theta, Y, Y_flag):
       
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_sum(tf.multiply(0.5,tf.multiply(tf.math.squared_difference(tf.matmul(X, theta), Y),Y_flag)))
    ### END CODE HERE ###
    
    return cost
tf.compat.v1.reset_default_graph()

with tf.compat.v1.Session() as sess:
    Y, Y_flag = create_placeholders(500)
    parameters = initialize_parameters(500, 600)
    cost = compute_cost(parameters["X_mat"],parameters["theta_mat"], Y, Y_flag)
    print("cost = " + str(cost))
def random_mini_batches(Y_mat, Y_flag, mini_batch_size = 64, seed = 0):
    
    m = Y_mat.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = Y_mat[:, permutation]
    shuffled_Y = Y_flag[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model(Y_mat, flag, learning_rate = 0.001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
    ops.reset_default_graph()                         
    tf.compat.v1.set_random_seed(1)                         
    seed = 3                                          
    (n_x, m) = Y_mat.shape                           
    costs = []                                       
    
    Y, Y_flag = create_placeholders(n_x)
    
    parameters = initialize_parameters(n_x, m)
    ### END CODE HERE ###
        
    cost = compute_cost(parameters["X_mat"],parameters["theta_mat"], Y, Y_flag)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.compat.v1.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.compat.v1.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                      
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(Y_mat, flag, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
               
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={Y: minibatch_X, Y_flag: minibatch_Y})
              
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                print(f"epoch_cost: {epoch_cost:{10}}")
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        
        return parameters
parameters = model(y_matrix, df_flag, minibatch_size = 671, num_epochs = 150)
parameters
results = np.dot(parameters["X_mat"],parameters["theta_mat"])
results.shape

rating_filled = pd.DataFrame(np.dot(parameters["X_mat"],parameters["theta_mat"]), columns = col_names, index=row_names)
rating_filled.head()
rating_filled[212][3269]
df.tail(30)
