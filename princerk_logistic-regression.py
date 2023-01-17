# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import ndimage, misc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import glob

import matplotlib.pyplot as plt

from random import shuffle



from PIL import Image

%matplotlib inline

# Any results you write to the current directory are saved as output.

trainlist = glob.glob('../input/train/*')



train_catlist = [imagename for imagename in trainlist if 'cat' in imagename]

train_doglist = [imagename for imagename in trainlist if 'dog' in imagename]



# take 200 samples of each cat and dogs

m_trainlist = train_catlist[:1000]

m_trainlist.extend(train_doglist[:1000])

shuffle(m_trainlist)



m_testlist = train_catlist[-200:]

m_testlist.extend(train_doglist[-200:])

shuffle(m_testlist)





# shuffle each train and test set



#trainlist = trainlist[:10]

#testlist = testlist[:10]

x_train_orig = np.array([np.array(misc.imresize(ndimage.imread(imagename, mode='RGB'), (256,256,3))) for imagename in m_trainlist])

x_test_orig = np.array([np.array(misc.imresize(ndimage.imread(imagename, mode='RGB'), (256,256,3))) for imagename in m_testlist])

y_train_orig = np.array([0 if 'cat' in imagename else 1 for imagename in m_trainlist]).reshape((-1, 1)).T

y_test_orig = np.array([0 if 'cat' in imagename else 1 for imagename in m_testlist]).reshape((-1, 1)).T

print(x_train_orig.shape)

print(x_test_orig.shape)

plt.imshow(x_train_orig[1])
x_train_flatten = x_train_orig.reshape(x_train_orig.shape[0], -1).T

x_test_flatten = x_test_orig.reshape(x_test_orig.shape[0], -1).T

print ("x_train_flatten shape: " + str(x_train_flatten.shape))

print ("y_train_orig shape: " + str(y_train_orig.shape))

print ("x_test_flatten shape: " + str(x_test_flatten.shape))

print ("y_test_orig shape: " + str(y_test_orig.shape))

print ("sanity check after reshaping: " + str(x_train_flatten[0:5,0]))

x_train = x_train_flatten / 255

x_test = x_test_flatten / 255

y_train = y_train_orig

y_test = y_test_orig
def sigmoid(z):

    """

    Arguments:

    z -- A scalar or numpy array of any size.



    Return:

    s -- sigmoid(z)

    """

    s = 1/(1 + np.exp(-z))

    return s
def initialize_with_zeros(dim):

    """

    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    

    Argument:

    dim -- size of the w vector we want (or number of parameters in this case)

    

    Returns:

    w -- initialized vector of shape (dim, 1)

    b -- initialized scalar (corresponds to the bias)

    """

    

    w = np.zeros((dim, 1))

    b = 0

    

    assert(w.shape == (dim, 1))

    assert(isinstance(b, float) or isinstance(b, int))

    

    return w, b
def propagate(w, b, X, Y):

    """

    Arguments:

    w -- weights, a numpy array of size (num_px * num_px * 3, 1)

    b -- bias, a scalar

    X -- data of size (num_px * num_px * 3, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if dog) of size (1, number of examples)



    Return:

    cost -- negative log-likelihood cost for logistic regression

    dw -- gradient of the loss with respect to w, thus same shape as w

    db -- gradient of the loss with respect to b, thus same shape as b

    """

    

    m = X.shape[1]

    

    # FORWARD PROPAGATION (FROM X TO COST)

   

    A = sigmoid(np.dot(w.T, X) + b)                                     # compute activation

    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

   

    

    # BACKWARD PROPAGATION (TO FIND GRAD)

   

    dw = 1/m * np.dot(X, (A -Y).T)

    db = 1/m * np.sum(A-Y)





    assert(dw.shape == w.shape)

    assert(db.dtype == float)

    cost = np.squeeze(cost)

    assert(cost.shape == ())

    

    grads = {"dw": dw,

             "db": db}

    

    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    """

    This function optimizes w and b by running a gradient descent algorithm

    

    Arguments:

    w -- weights, a numpy array of size (num_px * num_px * 3, 1)

    b -- bias, a scalar

    X -- data of shape (num_px * num_px * 3, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if dog), of shape (1, number of examples)

    num_iterations -- number of iterations of the optimization loop

    learning_rate -- learning rate of the gradient descent update rule

    print_cost -- True to print the loss every 100 steps

    

    Returns:

    params -- dictionary containing the weights w and bias b

    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function

    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    

    Tips:

    You basically need to write down two steps and iterate through them:

        1) Calculate the cost and the gradient for the current parameters. Use propagate().

        2) Update the parameters using gradient descent rule for w and b.

    """

    

    costs = []

    

    for i in range(num_iterations):

        

        

        # Cost and gradient calculation (≈ 1-4 lines of code)

        grads, cost = propagate(w, b, X, Y)

        

        # Retrieve derivatives from grads

        dw = grads["dw"]

        db = grads["db"]

        

        w = w - learning_rate * dw

        b = b - learning_rate * db

        

        # Record the costs

        if i % 100 == 0:

            costs.append(cost)

        

        # Print the cost every 100 training examples

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    

    params = {"w": w,

              "b": b}

    

    grads = {"dw": dw,

             "db": db}

    

    return params, grads, costs
def predict(w, b, X):

    '''

    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    

    Arguments:

    w -- weights, a numpy array of size (num_px * num_px * 3, 1)

    b -- bias, a scalar

    X -- data of size (num_px * num_px * 3, number of examples)

    

    Returns:

    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X

    '''

    

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    # Compute vector "A" predicting the probabilities of a cat being present in the picture

    ### START CODE HERE ### (≈ 1 line of code)

    A = sigmoid(np.dot(w.T, X) + b)

    ### END CODE HERE ###

    for i in range(A.shape[1]):

        

        # Convert probabilities A[0,i] to actual predictions p[0,i]

        ### START CODE HERE ### (≈ 4 lines of code)

        if A[0][i] > 0.5:

            Y_prediction[0][i] = 1

        ### END CODE HERE ###

    

    assert(Y_prediction.shape == (1, m))

    

    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    """

    Builds the logistic regression model by calling the function you've implemented previously

    

    Arguments:

    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)

    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)

    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)

    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)

    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters

    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()

    print_cost -- Set to true to print the cost every 100 iterations

    

    Returns:

    d -- dictionary containing information about the model.

    """

    

    ### START CODE HERE ###

    

    # initialize parameters with zeros (≈ 1 line of code)

    w, b = initialize_with_zeros(X_train.shape[0])



    # Gradient descent (≈ 1 line of code)

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    

    # Retrieve parameters w and b from dictionary "parameters"

    w = parameters["w"]

    b = parameters["b"]

    

    # Predict test/train set examples (≈ 2 lines of code)

    Y_prediction_test = predict(w, b, X_test)

    Y_prediction_train = predict(w, b, X_train)



    ### END CODE HERE ###



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))



    

    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train, 

         "w" : w, 

         "b" : b,

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
d = model(x_train, y_train, x_test, y_test, num_iterations = 2000, learning_rate = 0.001, print_cost = True)