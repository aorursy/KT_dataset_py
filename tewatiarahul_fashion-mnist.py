import numpy as np

import pandas as pd
train = pd.read_csv("../input/fashion-mnist_train.csv")
X_train_org = np.array(train.drop(['label'], axis=1)).reshape((-1, 28, 28, 1))
from keras.utils.np_utils import to_categorical

y_train_org = train.label

y_train_org = to_categorical(y_train_org, num_classes=10)
label_dictionary = {

    0: 'T_shirt_top',

    1: 'Trouser',

    2: 'Pullover',

    3: 'Dress',

    4: 'Coat',

    5: 'Sandal',

    6: 'Shirt',

    7: 'Sneaker',

    8: 'Bag',

    9: 'Ankle_boot',

}
#from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org, test_size = 0.01)

X_train = X_train_org

y_train = y_train_org

trainy = pd.read_csv("../input/fashion-mnist_test.csv")
X_val = np.array(trainy.drop(['label'], axis=1)).reshape((-1, 28, 28, 1))

y_val = trainy.label

y_val = to_categorical(y_val, num_classes=10)
import matplotlib.pyplot as plt

plt.imshow(X_train[90][:,:,0])
import math

import tensorflow as tf

from tensorflow.python.framework import ops
X_train = X_train/255.

X_test = X_val/255.

Y_train = y_train

Y_test = y_val

print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))

conv_layers = {}
def create_placeholders(n_H0, n_W0, n_C0, n_y):

    """

    Creates the placeholders for the tensorflow session.

    

    Arguments:

    n_H0 -- scalar, height of an input image

    n_W0 -- scalar, width of an input image

    n_C0 -- scalar, number of channels of the input

    n_y -- scalar, number of classes

        

    Returns:

    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"

    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"

    """



    ### START CODE HERE ### (≈2 lines)

    X = tf.placeholder(dtype = tf.float32, shape = (None, n_H0, n_W0, n_C0))

    Y = tf.placeholder(dtype = tf.float32, shape = (None, n_y))

    ### END CODE HERE ###

    

    return X, Y
X, Y = create_placeholders(64, 64, 1, 10)

print ("X = " + str(X))

print ("Y = " + str(Y))
def initialize_parameters():

    """

    Initializes weight parameters to build a neural network with tensorflow. The shapes are:

                        W1 : [4, 4, 1, 8]

                        W2 : [2, 2, 8, 16]

    Returns:

    parameters -- a dictionary of tensors containing W1, W2

    """

        

    ### START CODE HERE ### (approx. 2 lines of code)

    W1 = tf.get_variable("W1", [4, 4, 1, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer =  tf.contrib.layers.xavier_initializer(seed = 0))

    ### END CODE HERE ###



    parameters = {"W1": W1,

                  "W2": W2}

    

    return parameters
tf.reset_default_graph()

with tf.Session() as sess_test:

    parameters = initialize_parameters()

    init = tf.global_variables_initializer()

    sess_test.run(init)

    print("W1 = " + str(parameters["W1"].eval()[1,1,0]))

    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
def forward_propagation(X, parameters):

    """

    Implements the forward propagation for the model:

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "W2"

                  the shapes are given in initialize_parameters



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')

    

    Z1 = tf.layers.batch_normalization(Z1)

    

    A1 = tf.nn.relu(Z1)

    

    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    

    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')

    

    Z2 = tf.layers.batch_normalization(Z2)



    A2 = tf.nn.relu(Z2)

    

    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    

    P2 = tf.contrib.layers.flatten(P2)

    

    # 10 neurons in output layer. 

    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=10, activation_fn = None)

    

    #print(Z3.shape)

    print("Hello!")

    return Z3
tf.reset_default_graph()

with tf.Session() as sess:

    X, Y = create_placeholders(28, 28, 1, 10)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    init = tf.global_variables_initializer()

    sess.run(init)

    a = sess.run(Z3, {X: np.random.randn(2,28,28,1), Y: np.random.randn(2,10)})

    print("Z3 = " + str(a))
def compute_cost(Z3, Y):

    """

    Computes the cost

    

    Arguments:

    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)

    Y -- "true" labels vector placeholder, same shape as Z3

    

    Returns:

    cost - Tensor of the cost function

    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y))

    

    return cost
tf.reset_default_graph()



with tf.Session() as sess:

    np.random.seed(1)

    X, Y = create_placeholders(28, 28, 1, 10)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    init = tf.global_variables_initializer()

    sess.run(init)

    a = sess.run(cost, {X: np.random.randn(4,28,28,1), Y: np.random.randn(4,10)})

    print("cost = " + str(a))
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    """

    Creates a list of random minibatches from (X, Y)

    

    Arguments:

    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)

    mini_batch_size - size of the mini-batches, integer

    

    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """

    

    m = X.shape[0]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]

        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,

          num_epochs = 30, minibatch_size = 64, print_cost = True):

    """

    Implements a three-layer ConvNet in Tensorflow:

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    

    Arguments:

    X_train -- training set, of shape (None, 28, 28, 1)

    Y_train -- test set, of shape (None, n_y = 10)

    X_test -- training set, of shape (None, 28, 28, 1)

    Y_test -- test set, of shape (None, n_y = 10)

    learning_rate -- learning rate of the optimization

    num_epochs -- number of epochs of the optimization loop

    minibatch_size -- size of a minibatch

    print_cost -- True to print the cost every 100 epochs

    

    Returns:

    train_accuracy -- real number, accuracy on the train set (X_train)

    test_accuracy -- real number, testing accuracy on the test set (X_test)

    parameters -- parameters learnt by the model. They can then be used to predict.

    """

    

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables



    (m, n_H0, n_W0, n_C0) = X_train.shape             

    n_y = Y_train.shape[1]                            

    costs = []                                        # To keep track of the cost

    

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

   



    parameters = initialize_parameters()

    

    Z3 = forward_propagation(X, parameters)



    cost = compute_cost(Z3, Y)

    

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    optimizer = tf.group([optimizer, update_ops])

    init = tf.global_variables_initializer()

     

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):



            minibatch_cost = 0.

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)



            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _ , temp_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X,Y:minibatch_Y})

                

                minibatch_cost += temp_cost / num_minibatches

                



            # Print the cost every epoch

            if print_cost == True and epoch % 1 == 0:

                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            if print_cost == True and epoch % 1 == 0:

                costs.append(minibatch_cost)

        

        

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # Calculate the correct predictions

        predict_op = tf.argmax(Z3, 1)

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        print(str(predict_op))

        print((predict_op.shape))

        print(str(correct_prediction))

        print((correct_prediction.shape))

        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print(accuracy)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)

        print("Test Accuracy:", test_accuracy)

        '''

        while(True):

            i = input("Input image index: ")

            i = int(i)

            if(i != -1):

                sample = X_test[i].reshape(1,28,28,1)

                sample = sample.astype('float32')

                print("Actual image ==> "+label_dictionary[np.argmax(Y_test[i])] + ': ' + str(np.argmax(Y_test[i])))

                sample_1 = sample.reshape(28,28)

                plt.imshow(sample_1, cmap = 'Greys')

                pred= sess.run(Z3,{X:sample})

                a=tf.argmax(tf.nn.softmax(pred),1)

                print("Predicted image ==> "+ label_dictionary[a.eval()[0]] + ': ' + str(a.eval()[0]) )

                plt.show()

            else:

                print("It's over !!! End game")

                break

        '''   

        updated_params = {

            'W1':parameters['W1'].eval(),

            'W2':parameters['W2'].eval()

        }

        return train_accuracy, test_accuracy, updated_params
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)

_, _, updated_params = model(X_train, Y_train, X_test, Y_test)
