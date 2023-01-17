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
import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
yTrain = train.Target

xTrain = train.drop(['Target','Id'], axis=1)
yTrain.value_counts()
train_null = xTrain.isnull().sum()

train_null[train_null != 0]
xTrain.rez_esc.value_counts()
xTrain.loc[((xTrain['age'] > 19) | (xTrain['age'] < 7)) & (xTrain['rez_esc'].isnull()), 'rez_esc'] = 0

xTrain.rez_esc.fillna(xTrain.rez_esc.mean(), inplace=True)



xTrain.rez_esc.isnull().sum()
xTrain["v18q1"] = xTrain["v18q1"].fillna(0)
# Fill in households that own the house with 0 rent payment

xTrain.loc[(xTrain['tipovivi1'] == 1), 'v2a1'] = 0

xTrain["v2a1"].fillna(xTrain["v2a1"].mean(), inplace=True)

xTrain["meaneduc"].fillna(xTrain["meaneduc"].mean(), inplace=True)

xTrain["SQBmeaned"].fillna(xTrain["SQBmeaned"].mean(), inplace=True)
# Difference between people living in house and household size

xTrain['hhsize-diff'] = xTrain['tamviv'] - xTrain['hhsize']
xTrain['walls'] = np.argmax(np.array(xTrain[['epared1', 'epared2', 'epared3']]), axis = 1)



# Roof ordinal variable

xTrain['roof'] = np.argmax(np.array(xTrain[['etecho1', 'etecho2', 'etecho3']]), axis = 1)



# Floor ordinal variable

xTrain['floor'] = np.argmax(np.array(xTrain[['eviv1', 'eviv2', 'eviv3']]), axis = 1)



# Create new feature

xTrain['walls+roof+floor'] = xTrain['walls'] + xTrain['roof'] + xTrain['floor']
# No toilet, no electricity, no floor, no water service, no ceiling

xTrain['warning'] = 1 * (xTrain['sanitario1'] + 

                         xTrain['noelec'] + 

                         xTrain['pisonotiene'] + 

                         xTrain['abastaguano'] + 

                         (xTrain['cielorazo'] == 0))
# Owns a refrigerator, computer, tablet, and television

xTrain['bonus'] = 1 * (xTrain['refrig'] + 

                      xTrain['computer'] + 

                      (xTrain['v18q1'] > 0) + 

                      xTrain['television'])
# Per capita features

xTrain['phones-per-capita'] = xTrain['qmobilephone'] / xTrain['tamviv']

xTrain['tablets-per-capita'] = xTrain['v18q1'] / xTrain['tamviv']

xTrain['rooms-per-capita'] = xTrain['rooms'] / xTrain['tamviv']

xTrain['rent-per-capita'] = xTrain['v2a1'] / xTrain['tamviv']
xTrain['tech'] = xTrain['v18q'] + xTrain['mobilephone']
string_column = [f for f in xTrain.columns if xTrain.dtypes[f] == 'object']

string_column
xTrain.drop(['idhogar'], axis = 1, inplace = True)



xTrain['dependency'].replace('no', 0, inplace = True)

xTrain['edjefe'].replace('no', 0, inplace = True)

xTrain['edjefa'].replace('no', 0, inplace = True)



xTrain['dependency'].replace('yes', 1, inplace = True)

xTrain['edjefe'].replace('yes', 1, inplace = True)

xTrain['edjefa'].replace('yes', 1, inplace = True)
xTrain[['dependency','edjefe','edjefa']].head()
xTrain[['dependency','edjefe','edjefa']] = xTrain[['dependency','edjefe','edjefa']].apply(pd.to_numeric)



[f for f in xTrain.columns if xTrain.dtypes[f] == 'object']
pd.options.mode.use_inf_as_na = True
# labor force

xTrain['adult'] = xTrain['hogar_adul'] - xTrain['hogar_mayor']

xTrain['dependency_count'] = xTrain['hogar_nin'] + xTrain['hogar_mayor']

xTrain['dependency2'] = xTrain['dependency_count'] / xTrain['adult']

xTrain['child_percent'] = xTrain['hogar_nin']/xTrain['hogar_total']

xTrain['elder_percent'] = xTrain['hogar_mayor']/xTrain['hogar_total']

xTrain['adult_percent'] = xTrain['hogar_adul']/xTrain['hogar_total']
xTrain['rent_per_adult'] = xTrain['v2a1']/xTrain['hogar_adul']

xTrain['rent_per_person'] = xTrain['v2a1']/xTrain['hhsize']
# male-female ratio

xTrain['r4h1_percent_in_male'] = xTrain['r4h1'] / xTrain['r4h3']

xTrain['r4m1_percent_in_female'] = xTrain['r4m1'] / xTrain['r4m3']

xTrain['r4h1_percent_in_total'] = xTrain['r4h1'] / xTrain['hhsize']

xTrain['r4m1_percent_in_total'] = xTrain['r4m1'] / xTrain['hhsize']

xTrain['r4t1_percent_in_total'] = xTrain['r4t1'] / xTrain['hhsize']
# per capito

xTrain['rent_per_bedroom'] = xTrain['v2a1']/xTrain['bedrooms']

xTrain['edler_per_bedroom'] = xTrain['hogar_mayor']/xTrain['bedrooms']

xTrain['adults_per_bedroom'] = xTrain['adult']/xTrain['bedrooms']

xTrain['child_per_bedroom'] = xTrain['hogar_nin']/xTrain['bedrooms']

xTrain['male_per_bedroom'] = xTrain['r4h3']/xTrain['bedrooms']

xTrain['female_per_bedroom'] = xTrain['r4m3']/xTrain['bedrooms']

xTrain['bedrooms_per_person_household'] = xTrain['hhsize']/xTrain['bedrooms']
train_null = xTrain.isnull().sum()

train_null[train_null != 0]
# when there is no adult in the house, the result will be Nan

xTrain["rent_per_adult"] = xTrain["rent_per_adult"].fillna(0)



# similar to male/female ratio

xTrain["r4h1_percent_in_male"] = xTrain["r4h1_percent_in_male"].fillna(0)

xTrain["r4m1_percent_in_female"] = xTrain["r4m1_percent_in_female"].fillna(0)



# when there is no adult in the household, the value will be inf. So, we should fill in some very large number 

xTrain.dependency2 = xTrain.dependency2.fillna(1e5)
train_null = xTrain.isnull().sum()

train_null[train_null != 0]
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(xTrain, yTrain,stratify=yTrain, test_size = 0.15, random_state = 2, shuffle=True)

x_train.shape, y_train.shape, x_val.shape, y_val.shape
# credit to my homework from Coursera:

# Hyperparameter tuning, Regularization and Optimization



import math

import numpy as np

import pandas as pd

import h5py

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops





def softmax(x):

    if x.ndim == 2:

        x = x.T

        x = x - np.max(x, axis=0)

        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T



    x = x - np.max(x)  # address the overflow problem

    return np.exp(x) / np.sum(np.exp(x))





def initialize_parameters(dims):

    """

    Initializes parameters to build a neural network with tensorflow. The shapes are:



    Returns:

    parameters -- a dictionary of tensors containing weights and biases

    """

    tf.set_random_seed(1)

    parameters = {}

    L = len(dims) - 1  # input layer should not be counted

    for l in range(L - 1):

        parameters["W" + str(l + 1)] = tf.get_variable("W" + str(l + 1), [dims[l + 1], dims[l]],

                                                       initializer=tf.contrib.layers.xavier_initializer(seed=1))

        parameters["b" + str(l + 1)] = tf.get_variable("b" + str(l + 1), [dims[l + 1], 1],

                                                       initializer=tf.zeros_initializer())

    parameters["W" + str(L)] = tf.get_variable("W" + str(L), [dims[L], dims[L - 1]],

                                               initializer=tf.contrib.layers.xavier_initializer(seed=1))

    parameters["b" + str(L)] = tf.get_variable("b" + str(L), [dims[L], 1],

                                               initializer=tf.zeros_initializer())



    return parameters





def create_placeholders(n_x, n_y):

    """

    Creates the placeholders for the tensorflow session.



    """



    X = tf.placeholder(tf.float32, [n_x, None], name="X")

    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")



    return X, Y





def forward_propagation(X, parameters):

    """

    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX



    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

                  the shapes are given in initialize_parameters



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    L = len(parameters) // 2

    Z = {}

    A = {}

    A["0"] = X

    for l in range(L):

        Z[str(l + 1)] = tf.add(tf.matmul(parameters["W" + str(l + 1)], A[str(l)]), parameters["b" + str(l + 1)])

        A[str(l + 1)] = tf.nn.relu(Z[str(l + 1)])

    return Z[str(L)]





def compute_cost(ZL, Y, reg_constant=0.01):

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)

    logits = tf.transpose(ZL)

    labels = tf.transpose(Y)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)



    l2_loss = reg_constant * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

    c = loss + l2_loss

    cost = tf.reduce_mean(c)



    return cost





def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    """

    Creates a list of random minibatches from (X, Y)



    Arguments:

    X -- input data, of shape (input size, number of examples)

    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)

    mini_batch_size -- size of the mini-batches, integer



    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """



    np.random.seed(seed)  # To make your "random" minibatches the same as ours

    m = X.shape[1]  # number of training examples

    C = Y.shape[0]  # number of class

    mini_batches = []



    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X.values[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((C, m))



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(

        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]

        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)



    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, (mini_batch_size * num_complete_minibatches):]

        mini_batch_Y = shuffled_Y[:, (mini_batch_size * num_complete_minibatches):]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)



    return mini_batches





def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y





def compute_cost_with_regularization(AL, Y, parameters, lambd):

    """

    Implement the cost function with L2 regularization. See formula (2) above.



    Arguments:

    AL -- post-activation, output of forward propagation, of shape (output size, number of examples)

    Y -- "true" labels vector, of shape (output size, number of examples)

    parameters -- python dictionary containing parameters of the model



    Returns:

    cost - value of the regularized loss function (formula (2))

    """

    m = Y.shape[1]

    L = len(parameters) // 2



    cross_entropy_cost = compute_cost(AL, Y)  # This gives you the cross-entropy part of the cost



    sum = 0

    for l in range(L):

        sum += np.sum(np.square(parameters["W" + str(l + 1)]))



    L2_regularization_cost = lambd * sum / (2 * m)



    cost = cross_entropy_cost + L2_regularization_cost



    return cost





def backward_propagation_with_regularization(X, Y, parameters, Z, A, lambd):

    """

    Implements the backward propagation of our baseline model to which we added an L2 regularization.



    Arguments:

    X -- input dataset, of shape (input size, number of examples)

    Y -- "true" labels vector, of shape (output size, number of examples)

    cache -- cache output from forward_propagation()

    lambd -- regularization hyperparameter, scalar



    Returns:

    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables

    """



    m = X.shape[1]

    L = len(parameters) // 2

    dZ = {}

    dW = {}

    db = {}

    dA = {}



    dZ[str(L)] = A[str(L)] - Y



    dW[str(L)] = 1. / m * np.dot(dZ[str(L)], A[str(L - 1)].T) + (lambd * parameters["W" + str(L)]) / m

    db[str(L)] = 1. / m * np.sum(dZ[str(L)], axis=1, keepdims=True)



    for l in reversed(range(1, L)):

        dA[str(l)] = np.dot(parameters["W" + str(l + 1)].T, dZ[str(l + 1)])

        dZ[str(l)] = np.multiply(dA[str(l)], np.int64(A[str(l)] > 0))

        dW[str(l)] = 1. / m * np.dot(dZ[str(l)], A[str(l - 1)].T) + (lambd * parameters["W" + str(l)]) / m

        db[str(l)] = 1. / m * np.sum(dZ[str(l)], axis=1, keepdims=True)



    return dZ, dW, db, dA





def update_parameters_with_adam(parameters, dZ, dW, db, dA, v, s,

                                t, learning_rate, beta1, beta2, epsilon):

    """

    Update parameters using Adam



    Arguments:

    v -- Adam variable, moving average of the first gradient, python dictionary

    s -- Adam variable, moving average of the squared gradient, python dictionary

    learning_rate -- the learning rate, scalar.

    beta1 -- Exponential decay hyperparameter for the first moment estimates

    beta2 -- Exponential decay hyperparameter for the second moment estimates

    epsilon -- hyperparameter preventing division by zero in Adam updates



    Returns:

    parameters -- python dictionary containing your updated parameters

    v -- Adam variable, moving average of the first gradient, python dictionary

    s -- Adam variable, moving average of the squared gradient, python dictionary

    """



    L = len(parameters) // 2  # number of layers in the neural networks

    v_corrected = {}  # Initializing first moment estimate, python dictionary

    s_corrected = {}  # Initializing second moment estimate, python dictionary



    # Perform Adam update on all parameters

    for l in range(L):

        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".

        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * dW[str(l + 1)]

        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * db[str(l + 1)]



        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))

        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))



        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(dW[str(l + 1)], 2)

        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(db[str(l + 1)], 2)



        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))

        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))



        # Update parameters.

        # Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected[

            "dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected[

            "db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)



    return parameters, v, s





def model(dims, X_train, Y_train, X_test=None, Y_test=None, learning_rate=0.0001,

          reg=0.01, num_epochs=1500, minibatch_size=32, print_cost=True):

    """

    Implements a L-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->...->LINEAR->SOFTMAX.



    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """



    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    tf.set_random_seed(1)  # to keep consistent results

    seed = 3  # to keep consistent results

    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)

    n_y = Y_train.shape[0]  # n_y : output size, this is actually the number of class (C)

    costs = []  # To keep track of the cost



    # Create Placeholders of shape (n_x, n_y)

    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters

    parameters = initialize_parameters(dims)

    # Forward propagation: Build the forward propagation in the tensorflow graph

    ZL = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(ZL, Y, reg)

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



            epoch_cost = 0.  # Defines a cost related to an epoch

            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:

                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch



                # IMPORTANT: The line that runs the graph on a minibatch.

                # Run the session to execute the "optimizer" and the "cost",

                # the feedict should contain a minibatch for (X,Y).

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})



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

        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        if Y_test is not None:

            print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))



        return parameters





def predict(X, parameters):

    x = tf.placeholder("float", [X.shape[0], X.shape[1]])



    ZL = forward_propagation(x, parameters)

    p = tf.argmax(ZL)



    sess = tf.Session()

    prediction = sess.run(p, feed_dict={x: X})



    return prediction





def one_hot_matrix(labels, C):

    """

    Creates a matrix where the i-th row corresponds to the ith class number and the jth column

                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)

                     will be 1.



    Arguments:

    labels -- vector containing the labels

    C -- number of classes, the depth of the one hot dimension



    Returns:

    one_hot -- one hot matrix

    """



    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)

    C = tf.constant(C, name='C')



    # Use tf.one_hot, be careful with the axis (approx. 1 line)

    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)



    # Create the session (approx. 1 line)

    sess = tf.Session()



    # Run the session (approx. 1 line)

    one_hot = sess.run(one_hot_matrix)



    # Close the session (approx. 1 line). See method 1 above.

    sess.close()



    return one_hot

Y_train = convert_to_one_hot(np.array(y_train), 5)

Y_train.shape
X_train = x_train.T

X_train.shape
Y_val = convert_to_one_hot(np.array(y_val), 5)

X_val = x_val.T

Y_val.shape, X_val.shape
### CONSTANTS DEFINING THE MODEL ####

n_x = X_train.shape[0]

n_h1 = 200  # hyper parameter

n_h2 = 200  # hyper parameter

n_h3 = 100  # hyper parameter

n_y = Y_train.shape[0]

layers_dims = [n_x, n_h1, n_h2, n_h3, n_y]

layers_dims
parameters = model(layers_dims, X_train, Y_train, X_val, Y_val, learning_rate = 0.001, reg = 0, num_epochs = 1500)
xTest = test.drop(['Id'], axis=1)
test_null = xTest.isnull().sum()

test_null[test_null != 0]
xTest["v18q1"] = xTest["v18q1"].fillna(0)
xTest.loc[((xTest['age'] > 19) | (xTest['age'] < 7)) & (xTest['rez_esc'].isnull()), 'rez_esc'] = 0

xTest.rez_esc.fillna(xTest.rez_esc.mean(), inplace=True)



xTest.rez_esc.isnull().sum()
# Fill in households that own the house with 0 rent payment

xTest.loc[(xTest['tipovivi1'] == 1), 'v2a1'] = 0

xTest["v2a1"].fillna(xTest["v2a1"].mean(), inplace=True)

xTest["meaneduc"].fillna(xTest["meaneduc"].mean(), inplace=True)

xTest["SQBmeaned"].fillna(xTest["SQBmeaned"].mean(), inplace=True)
# Difference between people living in house and household size

xTest['hhsize-diff'] = xTest['tamviv'] - xTest['hhsize']
xTest['walls'] = np.argmax(np.array(xTest[['epared1', 'epared2', 'epared3']]), axis = 1)



# Roof ordinal variable

xTest['roof'] = np.argmax(np.array(xTest[['etecho1', 'etecho2', 'etecho3']]), axis = 1)



# Floor ordinal variable

xTest['floor'] = np.argmax(np.array(xTest[['eviv1', 'eviv2', 'eviv3']]), axis = 1)



# Create new feature

xTest['walls+roof+floor'] = xTest['walls'] + xTest['roof'] + xTest['floor']
# No toilet, no electricity, no floor, no water service, no ceiling

xTest['warning'] = 1 * (xTest['sanitario1'] + 

                         xTest['noelec'] + 

                         xTest['pisonotiene'] + 

                         xTest['abastaguano'] + 

                         (xTest['cielorazo'] == 0))
# Owns a refrigerator, computer, tablet, and television

xTest['bonus'] = 1 * (xTest['refrig'] + 

                      xTest['computer'] + 

                      (xTest['v18q1'] > 0) + 

                      xTest['television'])
# Per capita features

xTest['phones-per-capita'] = xTest['qmobilephone'] / xTest['tamviv']

xTest['tablets-per-capita'] = xTest['v18q1'] / xTest['tamviv']

xTest['rooms-per-capita'] = xTest['rooms'] / xTest['tamviv']

xTest['rent-per-capita'] = xTest['v2a1'] / xTest['tamviv']
xTest['tech'] = xTest['v18q'] + xTest['mobilephone']
test_null = xTest.isnull().sum()

test_null[test_null != 0]
xTest = xTest.drop(['idhogar'], axis = 1)



xTest['dependency'].replace('no', 0, inplace = True)

xTest['edjefe'].replace('no', 0, inplace = True)

xTest['edjefa'].replace('no', 0, inplace = True)



xTest['dependency'].replace('yes', 1, inplace = True)

xTest['edjefe'].replace('yes', 1, inplace = True)

xTest['edjefa'].replace('yes', 1, inplace = True)
# labor force

xTest['adult'] = xTest['hogar_adul'] - xTest['hogar_mayor']

xTest['dependency_count'] = xTest['hogar_nin'] + xTest['hogar_mayor']

xTest['dependency2'] = xTest['dependency_count'] / xTest['adult']

xTest['child_percent'] = xTest['hogar_nin']/xTest['hogar_total']

xTest['elder_percent'] = xTest['hogar_mayor']/xTest['hogar_total']

xTest['adult_percent'] = xTest['hogar_adul']/xTest['hogar_total']
xTest['rent_per_adult'] = xTest['v2a1']/xTest['hogar_adul']

xTest['rent_per_person'] = xTest['v2a1']/xTest['hhsize']
# male-female ratio

xTest['r4h1_percent_in_male'] = xTest['r4h1'] / xTest['r4h3']

xTest['r4m1_percent_in_female'] = xTest['r4m1'] / xTest['r4m3']

xTest['r4h1_percent_in_total'] = xTest['r4h1'] / xTest['hhsize']

xTest['r4m1_percent_in_total'] = xTest['r4m1'] / xTest['hhsize']

xTest['r4t1_percent_in_total'] = xTest['r4t1'] /xTest['hhsize']
# per capito

xTest['rent_per_bedroom'] = xTest['v2a1']/xTest['bedrooms']

xTest['edler_per_bedroom'] = xTest['hogar_mayor']/xTest['bedrooms']

xTest['adults_per_bedroom'] = xTest['adult']/xTest['bedrooms']

xTest['child_per_bedroom'] = xTest['hogar_nin']/xTest['bedrooms']

xTest['male_per_bedroom'] = xTest['r4h3']/xTest['bedrooms']

xTest['female_per_bedroom'] = xTest['r4m3']/xTest['bedrooms']

xTest['bedrooms_per_person_household'] = xTest['hhsize']/xTest['bedrooms']
# when there is no adult in the house, the result will be Nan

xTest["rent_per_adult"] = xTest["rent_per_adult"].fillna(0)



# similar to male/female ratio

xTest["r4h1_percent_in_male"] = xTest["r4h1_percent_in_male"].fillna(0)

xTest["r4m1_percent_in_female"] = xTest["r4m1_percent_in_female"].fillna(0)



# when there is no adult in the household, the value will be inf. So, we should fill in some very large number 

xTest.dependency2 = xTest.dependency2.fillna(1e5)
test_null = xTest.isnull().sum()

test_null[test_null != 0]
X_test = xTest.T

X_test.shape
prediction = predict(X_test, parameters)
pd.Series(prediction).value_counts()
prediction[prediction==0] = 1
test['Target'] = prediction

df2 = pd.DataFrame({'Id':test['Id'],'Target':test.Target})

df2.to_csv("submit.csv",index=False)