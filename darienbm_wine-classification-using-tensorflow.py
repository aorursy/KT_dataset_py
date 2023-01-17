from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np 

import pandas as pd 

import math

import matplotlib.pyplot as plt # we will use this for plotting

from sklearn.model_selection import train_test_split # very handy way of selecting sets

import tensorflow as tf
data = pd.read_csv('../input/Wine.csv', header=None)

data.columns = ['name'

                ,'alcohol'

                ,'malicAcid'

                ,'ash'

                ,'ashalcalinity'

                 ,'magnesium'

                ,'totalPhenols'

                 ,'flavanoids'

                 ,'nonFlavanoidPhenols'

                 ,'proanthocyanins'

                ,'colorIntensity'

                 ,'hue'

                 ,'od280_od315'

                 ,'proline'

                ]



data.isnull().sum() # Check if there are any missing values
import seaborn as sns

correlations = data[data.columns].corr(method='pearson')

sns.heatmap(correlations, cmap="YlGnBu", annot = True)
import heapq



print('Absolute overall correlations')

print('-' * 30)

correlations_abs_sum = correlations[correlations.columns].abs().sum()

print(correlations_abs_sum, '\n')



print('Weakest correlations')

print('-' * 30)

print(correlations_abs_sum.nsmallest(3))
# From this we learn that we could drop these 3 parameters and improve our algorithm...possibly?

# We also need to drop 'name' as that is our label vector in fact!

#X_data = data.drop(['name','ash', 'magnesium', 'colorIntensity'], axis=1)

#X_data = data.drop(['name','ash', 'magnesium'], axis=1)

X_data = data.drop(['name','ash'], axis=1)



Y_data = data.iloc[:,:1] # take all the names (see pandas reference for iloc vs loc)

classes = Y_data.name.unique()

num_classes = len(classes)

print('Class names: ', classes)

print('Number of classes: ', num_classes)
X = X_data.values 

Y = Y_data.values

print('Data types: ', type(X), type(Y))
def labelMaker(val):

    if val == 1:

        return [1, 0, 0]

    elif val == 2:

        return [0, 1, 0]

    else: 

        return [0, 0, 1]



Y = np.array([labelMaker(i[0]) for i in Y])

print(Y.shape)
from sklearn.model_selection import train_test_split # very handy way of selecting training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)



training_size = X_train.shape[0]

num_parameters = X_train.shape[1]

num_classes = Y_train.shape[1]

print('Training size: ', training_size)

print('Number of parameters: ', num_parameters)

print('Number of classes: ', num_classes)

# (n_x: input size, m : number of examples in the train set)

# (n_y : output size, m: number of examples)

X_train = X_train.transpose()

Y_train = Y_train.transpose()

X_test = X_test.transpose()

Y_test = Y_test.transpose()



print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
# n_x = num__input_features

# n_y = expected output (num classes)

def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, [n_x, None], name="X")

    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y



def initialize_parameters(num_input_features=12):

    """

    Initializes parameters to build a neural network with tensorflow. The shapes are:

                        W1 : [num_hidden_layer, num_input_features]

                        b1 : [num_hidden_layer, 1]

                        W2 : [num_output_layer_1, num_hidden_layer]

                        b2 : [num_output_layer_1, 1]

                        W3 : [num_output_layer_2, num_output_layer_1]

                        b3 : [num_output_layer_2, 1]

    

    Returns:

    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3

    """ 

    tf.set_random_seed(1)           

    W1 = tf.get_variable("W1", [10, num_input_features], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b1 = tf.get_variable("b1", [10, 1], initializer = tf.zeros_initializer())

    W2 = tf.get_variable("W2", [5, 10], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b2 = tf.get_variable("b2", [5, 1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [3, 5], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b3 = tf.get_variable("b3", [3, 1], initializer = tf.zeros_initializer())

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W3": W3,

                  "b3": b3}

    

    return parameters



def forward_propagation(X, parameters):

    """

    Implements the forward propagation for the model: 

    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters 

    "W1", "b1", "W2", "b2", "W3", "b3"

    the shapes are given in initialize_parameters



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:

    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1

    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2

    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

    ### END CODE HERE ###

    

    '''

     It is important to note that the forward propagation stops at z3. 

     The reason is that in tensorflow the last linear layer output is 

     given as input to the function computing the loss. 

     Therefore, you don't need a3!

    '''

    return Z3



def compute_cost(Z3, Y):

    """

    Computes the cost

    

    Arguments:

    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (3, number of examples)

    Y -- "true" labels vector placeholder, same shape as Z3

    

    Returns:

    cost - Tensor of the cost function

    """

    

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

   

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # The newer recommended function in Tensor flow

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return cost
num__input_features = 12

num_output_features = 3 # (num classes)



tf.reset_default_graph()



with tf.Session() as sess:

    X, Y = create_placeholders(num__input_features, num_output_features)

    parameters = initialize_parameters(num__input_features)

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    print("cost = " + str(cost))
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    """

    Creates a list of random minibatches from (X, Y)

    

    Arguments:

    X -- input data, of shape (input size, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    mini_batch_size - size of the mini-batches, integer

    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    

    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """

    

    m = X.shape[1]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))



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

from tensorflow.python.framework import ops



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,

          num_epochs = 1500, minibatch_size = 32, print_cost = True):

    

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

    tf.set_random_seed(1)                             # to keep consistent results

    seed = 3                                          # to keep consistent results

    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    n_y = Y_train.shape[0]                            # n_y : output size

    costs = []                                        # To keep track of the cost

    

    # Create Placeholders of shape (n_x, n_y)

    X, Y = create_placeholders(n_x, n_y)



    # Initialize parameters

    parameters = initialize_parameters(12)

    

    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z3 = forward_propagation(X, parameters)

    

    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(Z3, Y)

    

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    

    # Initialize all the variables

    init = tf.global_variables_initializer()



    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Run the initialization

        sess.run(init)

        

        # Do the training loop

        for epoch in range(num_epochs):



            epoch_cost = 0.                       # Defines a cost related to an epoch

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:



                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch

                

                # IMPORTANT: The line that runs the graph on a minibatch.

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).

                ### START CODE HERE ### (1 line)

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                ### END CODE HERE ###

                

                epoch_cost += minibatch_cost / num_minibatches



            # Print the cost every epoch

            if print_cost == True and epoch % 100 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

                

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # lets save the parameters in a variable

        parameters = sess.run(parameters)

        print("Parameters have been trained!")



        # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        

        return parameters
#parameters = model(X_train, Y_train, X_test, Y_test) # This is before I played with dev set

parameters = model(X_train, Y_train, X_test, Y_test, minibatch_size = 2) # This is the optimal I found

'''

Use train_test_split from sklearn.model_selection to split the data in dev and test sets

We will use a distribution of 60% for training, and 20% for both dev and test

'''



X = X_data.values 

Y = Y_data.values



# Recreate labels as hot-encoded using our function

Y = np.array([labelMaker(i[0]) for i in Y])



# Split first 60% vs 40%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=0)



# Split then that 40% of the test set in 50/50

X_dev, X_test, Y_dev, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)



# We transpose all the matrixes to compile with expected shapes

X_train = X_train.transpose()

X_test = X_test.transpose()

X_dev = X_dev.transpose()

Y_train = Y_train.transpose()

Y_test = Y_test.transpose()

Y_dev = Y_dev.transpose()



print(X_train.shape, X_test.shape, X_dev.shape)

print(Y_train.shape, Y_test.shape, Y_dev.shape)



# We feed our data 

parameters = model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 2)



# We feed our data 

parameters = model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 2)
