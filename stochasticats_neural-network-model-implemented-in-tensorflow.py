import math

import pandas as pd

import numpy as np 

import tensorflow as tf

import matplotlib.pyplot as plt

import scipy

from tensorflow.python.framework import ops



%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'
dataset = pd.read_csv("../input/creditcard.csv")
dataset.head()
dataset.describe()
print("Percent of total transactions that are fraudulent")

print(dataset["Class"].mean()*100)
print("Losses due to fraud:")

print("Total amount lost to fraud")

print(dataset.Amount[dataset.Class == 1].sum())

print("Mean amount per fraudulent transaction")

print(dataset.Amount[dataset.Class == 1].mean())

print("Compare to normal transactions:")

print("Total amount from normal transactions")

print(dataset.Amount[dataset.Class == 0].sum())

print("Mean amount per normal transactions")

print(dataset.Amount[dataset.Class == 0].mean())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 40



ax1.hist(dataset.Amount[dataset.Class == 1], bins = bins, normed = True, alpha = 0.75, color = 'red')

ax1.set_title('Fraud')



ax2.hist(dataset.Amount[dataset.Class == 0], bins = bins, normed = True, alpha = 0.5, color = 'blue')

ax2.set_title('Not Fraud')



plt.xlabel('Amount')

plt.ylabel('% of Transactions')

plt.yscale('log')

plt.show()
bins = 75

plt.hist(dataset.Time[dataset.Class == 1], bins = bins, normed = True, alpha = 0.75, label = 'Fraud', color = 'red')

plt.hist(dataset.Time[dataset.Class == 0], bins = bins, normed = True, alpha = 0.5, label = 'Not Fraud', color = 'blue')

plt.legend(loc='upper right')

plt.xlabel('Time (seconds)')

plt.ylabel('% of ')

plt.title('Transactions over Time')

plt.show()
Vfeatures = dataset.iloc[:,1:29].columns

print(Vfeatures)
import matplotlib.gridspec as gridspec

import seaborn as sns

bins = 50

plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, V in enumerate(dataset[Vfeatures]):

    ax = plt.subplot(gs[i])

    sns.distplot(dataset[V][dataset.Class == 1], bins = bins, norm_hist = True, color = 'red')

    sns.distplot(dataset[V][dataset.Class == 0], bins = bins, norm_hist = True, color = 'blue')

    ax.set_xlabel('')

    ax.set_title('distributions (w.r.t fraud vs. non-fraud) of feature: ' + str(V))

plt.show()
model_features = dataset.iloc[:,1:30].columns

print(model_features)



# normalize Amount column

dataset["Amount"] = (dataset["Amount"]-dataset["Amount"].mean())/dataset["Amount"].std()



# shuffle and split our data set

dataset = dataset.sample(frac=1).reset_index(drop=True)

split = np.random.rand(len(dataset)) < 0.95

dataset_train = dataset[split]

dataset_test = dataset[~split]

train_x = dataset_train.as_matrix(columns = model_features)

train_y = dataset_train["Class"]

test_x = dataset_test.as_matrix(columns = model_features)

test_y = dataset_test["Class"]



# check the distribution of fraud between train and test

# if these are too far off, try shuffling again

print(dataset["Amount"].sum())

print(train_y.mean()*100)

print(test_y.mean()*100)
''' 

modify train and test sets for correct dimensions and check it.

dimensions should be 

X - (# of features, # of examples)

Y - (1, # of examples)

'''

train_x = train_x.T

train_y = np.reshape(train_y, (1,len(dataset_train)))

test_x = test_x.T

test_y = np.reshape(test_y, (1,len(dataset_test)))



print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
def create_placeholders(n_x, n_y):

    # n_x - number of features

    # n_y - number of classes

    X = tf.placeholder(tf.float32, shape = (n_x, None))

    Y = tf.placeholder(tf.float32, shape = (n_y, None))

    return X, Y
def initialize_parameters():                  

    W1 = tf.get_variable("W1", [14,29], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

    b1 = tf.get_variable("b1", [14,1], initializer = tf.zeros_initializer())

    W2 = tf.get_variable("W2", [7,14], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

    b2 = tf.get_variable("b2", [7,1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [1,7], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())



    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W3": W3,

                  "b3": b3}

    

    return parameters
def forward_propagation(X, parameters):

    # Retrieve the parameters from the dictionary parameters

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']



    Z1 = tf.add(tf.matmul(W1, X), b1)                                   

    A1 = tf.nn.elu(Z1)                                              

    Z2 = tf.add(tf.matmul(W2, A1), b2)                                      

    A2 = tf.nn.elu(Z2)                                         

    Z3 = tf.add(tf.matmul(W3, A2), b3)

    

    return Z3
def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

    

    return cost
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    

    m = X.shape[1]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    

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
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,

          num_epochs = 1500, minibatch_size = 1024, print_cost = True):

    # Implements a three layer layer neural network using tensorflow

    

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

    tf.set_random_seed(1)                             # to keep consistent results

    seed = 3                                          # to keep consistent results

    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    n_y = Y_train.shape[0]                            # n_y : output size

    costs = []                                        # To keep track of the cost



    X, Y = create_placeholders(n_x, n_y)



    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    

    init = tf.global_variables_initializer()



    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):



            epoch_cost = 0.                       # Defines a cost related to an epoch

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

                

            if print_cost == True and epoch % 100 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

                

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        parameters = sess.run(parameters)

        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))



        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        

        return parameters
model_params = model(train_x, train_y, test_x, test_y)