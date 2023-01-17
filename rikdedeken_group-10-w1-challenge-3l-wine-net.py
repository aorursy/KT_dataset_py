# Package imports
# Matplotlib is a matlab like plotting library

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
# Numpy handles matrix operations
import numpy as np
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
# Display plots inline and change default figure size
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

df = pd.read_csv('../input/W1data.csv')
df.head()
# Get labels
y = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values
# Get inputs
X = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis = 1)
X = X.values # we want a Numpy array, not a Pandas DataFrame
print (type(X))
X.shape, y.shape # Print shapes just to check
def softmax(z):
    '''
    Calculates the softmax activation of a given input x
    See: https://en.wikipedia.org/wiki/Softmax_function
    '''
    #Calculate exponent term first
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
def softmax_loss(y,y_hat):
    '''
    Calculates the generalized logistic loss between a prediction y_hat and the labels y
    See: http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

    We need to clip values that get too close to zero to avoid zeroing out. 
    Zeroing out is when a number gets so small that the computer replaces it with 0.
    Therefore, we clip numbers to a minimum value.
    '''
    # Clipping calue
    minval = 0.000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
    return loss
# Log loss derivative, equal to softmax loss derivative
def loss_derivative(y,y_hat):
    '''
    Calculates the gradient (derivative) of the loss between point y and y_hat
    See: https://stats.stackexchange.com/questions/219241/gradient-for-logistic-loss-function
    '''
    return (y_hat-y)
def tanh_derivative(x):
    '''
    Calculates the derivative of the tanh function that is used as the first activation function
    See: https://socratic.org/questions/what-is-the-derivative-of-tanh-x
    '''
    return (1 - np.power(x, 2))
def forward_prop(model,a0):
    '''
    Forward propagates through the model, stores results in cache.
    See: https://stats.stackexchange.com/questions/147954/neural-network-forward-propagation
    A0 is the activation at layer zero, it is the same as X
    '''
    ##done: add W/b/a/z for layer 3, change act. functions for L2 and L3
    
    # Load parameters from model
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']
    
    # Linear step
    z1 = (a0.dot(W1))
    z1 += b1
    # First activation function
    a1 = np.tanh(z1)
    
    # Second linear step
    z2 = a1.dot(W2)
    z2 += b2
    # Second activation function
    a2 = np.tanh(z2)
    
    # Third linear step
    z3 = a2.dot(W3)
    z3 += b3
    # Third activation function
    a3 = softmax(z3)
    cache = {        'a0':a0,
             'z1':z1,'a1':a1,
             'z2':z2,'a2':a2,
             'z3':z3,'a3':a3
            }
    return cache
def backward_prop(model,cache,y):
    '''
    Backward propagates through the model to calculate gradients.
    Stores gradients in grads dictionary.
    See: https://en.wikipedia.org/wiki/Backpropagation
    '''
    
    ##done: add W/b/a, dW/db/dz for layer 3, change derivations for L2/L3

    # Load parameters from model
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']

    # Load forward propagation results
    a0 = cache['a0']
    a1 = cache['a1']
    a2 = cache['a2']
    a3 = cache['a3']

    # Get number of samples
    m = y.shape[0]
    
    # Backpropagation
    # Calculate loss derivative with respect to L3 linears
    dz3 = loss_derivative(y=y,y_hat=a3)
    # Calculate loss derivative with respect to L3 weights
    dW3 = 1/m*(a2.T).dot(dz3)
    # Calculate loss derivative with respect to L3 bias
    db3 = 1/m*np.sum(dz3, axis=0)

    # Calculate loss derivative with respect to L2 linears
    dz2 = np.multiply(dz3.dot(W3.T), tanh_derivative(a2))
    # Calculate loss derivative with respect to L2 weights
    dW2 = 1/m*(a1.T).dot(dz2)
    # Calculate loss derivative with respect to L2 bias
    db2 = 1/m*np.sum(dz2, axis=0)
    
    # Calculate loss derivative with respect to L1 linears
    dz1 = np.multiply(dz2.dot(W2.T), tanh_derivative(a1))
    # Calculate loss derivative with respect to L1 weights
    dW1 = 1/m*np.dot(a0.T, dz1)
    # Calculate loss derivative with respect to L1 bias
    db1 = 1/m*np.sum(dz1, axis=0)
    
    # Store gradients
    grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2,
            'dW3': dW3,
            'db3': db3
            }
    return grads
#def initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim):
def initialize_parameters(dim_L0, dim_L1, dim_L2, dim_L3):

    '''
    Initializes weights with random number between -1 and 1
    Initializes bias with 0
    Assigns weights and parameters to model
    '''
    ##done: accept extra hdim for 3rd layer, add W/b for layer 3

    # L1 weights
    W1 = 2 * np.random.randn(dim_L0, dim_L1) - 1
    # L1 bias
    b1 = np.zeros((1, dim_L1))
    
    # L2 weights
    W2 = 2 * np.random.randn(dim_L1, dim_L2) - 1
    # L2 bias
    b2 = np.zeros((1, dim_L2))
    
    # L3 weights
    W3 = 2 * np.random.randn(dim_L2, dim_L3) - 1
    # L3 bias
    b3 = np.zeros((1, dim_L3))

    # Package and return model
    model = {
        'W1': W1, 
        'b1': b1, 
        'W2': W2, 
        'b2': b2,
        'W3': W3, 
        'b3': b3
    }
    return model
def update_parameters(model,grads,learning_rate):
    '''
    Updates parameters accoarding to gradient descent algorithm
    See: https://en.wikipedia.org/wiki/Gradient_descent
    '''
    ##done: add W/b for layer 3

    # Load parameters
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']
    
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    W3 -= learning_rate * grads['dW3']
    b3 -= learning_rate * grads['db3']

    # Store and return parameters
    model = {
        'W1': W1, 
        'b1': b1, 
        'W2': W2, 
        'b2': b2,
        'W3': W3, 
        'b3': b3,
    }
    return model
def predict(model, x):
    '''
    Predicts y_hat as 1 or 0 for a given input X
    '''
    ##done change a2 into a3
    
    # Do forward pass
    c = forward_prop(model,x)
    #get y_hat
    y_hat = np.argmax(c['a3'], axis=1)
    return y_hat
def train(model,X_,y_,learning_rate, stopatloss=0, stopatacc=0, epochs=20000, print_loss=False):
    '''
    Trains model
    Code adapted by Rik to accomodate training stop at desired Loss or Accuracy.
    Also changed the print_loss logic a bit so that long trainings do not run out of screen space.
    '''
    # Gradient descent. Loop over epochs
    stop = False
    epochs = round(epochs,2)
    L = 1
    acc = 0
    
    for i in range(0, epochs):
        if stop:
            break
        # Forward propagation
        cache = forward_prop(model,X_)

        # Backpropagation
        grads = backward_prop(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
    
        ##done change a2 into a3
        
        # Print loss & accuracy every 100 iterations
        if i % 10**2 == 0:
            # calculate Loss
            a3 = cache['a3']
            L = softmax_loss(y_,a3)
            y_hat = predict(model,X_)
            y_true = y_.argmax(axis=1)
            acc = accuracy_score(y_pred=y_hat,y_true=y_true)*100
        if print_loss and i % 10**2 == 0:
            print('Loss after iteration',i,':',L)
            print('Accuracy after iteration',i,':',acc,'%')
        if stopatloss and L <= stopatloss:
            print('Loss at stop-threshold of',stopatloss)
            stop = True
        if stopatacc and acc >= stopatacc:
            print('Accuracy at stop-threshold of',stopatacc,'%')
            stop = True

    if not print_loss:
        print('Loss after iteration',i,':',L)
        print('Accuracy after iteration',i,':',acc,'%')
    return model
# Hyper parameters

##done introduce hidden_layer_size for 3rd layer
size_L1 = 8
size_L2 = 5
# Total NN is now dimensioned as 13 -> size_L1 -> size_L2 -> 3

learning_rate = .5
# Initialize the parameters to random values. We need to learn these.
np.random.seed(0) # was 0
# This is what we return at the end
model = initialize_parameters(dim_L0=13, dim_L1=size_L1, dim_L2=size_L2, dim_L3=3)
# model = train(model,X,y,learning_rate=learning_rate,epochs=100000,print_loss=False, stopatacc=99)
model = train(model,X,y,learning_rate=learning_rate,epochs=100000,print_loss=False, stopatloss=.0001)
# use the output of this print to re-create a trained model
# print(model)