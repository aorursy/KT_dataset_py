# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load dataset
data_original = pd.read_csv(os.path.join(dirname, filename))
data_original.head()

# Select columns
data_select = data_original[['0', '1', '2', '1558']]
data_select.head()

# Remove NAs from any column
#data_select = data_select[data_select[['0','1','2','1558']].notna()]
data_select = data_select[data_select['0'].notna()]
data_select = data_select[data_select['1'].notna()]
data_select = data_select[data_select['2'].notna()]
data_select = data_select[data_select['1558'].notna()]

# Determine data type for each column
#data_select[['0', '1', '2']] = data_select[['0', '1', '2']].astype(float)

data_select['0'] = pd.to_numeric(data_select['0'], errors = 'coerce')
data_select['1'] = pd.to_numeric(data_select['1'], errors = 'coerce')
data_select['2'] = pd.to_numeric(data_select['2'], errors = 'coerce')
data_select[['1558']] = data_select[['1558']].astype('string')

# Identify frequency of ads and non-ads
data_select.groupby('1558').count()

# Convert last column to 1 or 0. 1 is ad, 0 is non-ad
data_select['1558'] = np.where(data_select['1558'] == 'ad.', 1, 0)
data_select['1558'] = data_select['1558'].astype(float)

# Remove NaN
data_select = data_select.dropna()

# Rename columns
data_select.columns = ['height', 'width', 'aspect_ratio', 'category']
#data_select.head()
#data_select.tail()
#data_select.dtypes

# Split data into train and development sets
random.seed(1)
data_select = shuffle(data_select)

train_set, test_set = train_test_split(data_select, train_size=1895, test_size=(2369-1895))

X_train = train_set[['height', 'width', 'aspect_ratio']]
Y_train = train_set[['category']]

X_test = test_set[['height', 'width', 'aspect_ratio']]
Y_test = test_set[['category']]

X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T

# Convert to numpy array
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

# Create logistic regression with neural network framework
# List of helper functions
# sigmoid()
# initialise_with_zeros()
# propagate()
# optimise()
# predict()
# model()

# sigmoid()
def sigmoid(z):
    # Compute the sigmoid of z
    # Arguments:
    # z - A scalar or numpy array of any size
    # Return:
    # s - sigmoid(z)
    s = 1 / (1 + np.exp(-z))
    return s
# Test sigmoid()
print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))
# initialise_with_zeros()
def initialise_with_zeros(dim):
    # This function creates a vector of zeros of shape (dim, 1) for w and initialises b to zero.
    # Arguments:
    # dim - size of the vector w we want (or number of parameters in this case)
    # Return:
    # w - initialised vector of shape (dim, 1)
    # b - initialised scalar (corresponds to the bias)
    w = np.zeros(shape=(dim, 1))
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
# Test initialise_with_zeros()
dim = 3
w, b = initialise_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))
# propagate()
def propagate(w, b, X, Y):
    # Implement the cost function and its gradient
    # Arguments:
    # w - weights, a numpy array of size (num_px, 1)
    # b - bias, a scalar
    # X - data of size (num_px, number of examples m)
    # Y - true "label" vector of size (1, number of examples m). 1 is ad and 0 is non-ad.
    # Return
    # cost - negative log-likelihood cost for logistic regression
    # dw - gradient of the loss with respect to w, thus same shape as w
    # db - gradient of the loss with respect to b, thus same shape as b
    m = X.shape[1]
    
    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    
    # Compute cost
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    
    # Backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {
        "dw" : dw,
        "db" : db
    }

    return grads, cost

# Test propagate()
# w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
# w, b, X, Y = np.array([[1.], [2.], [1.2]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2], [1., -1.1, 2.]]), np.array([[1, 0, 1]])
w, b, X, Y = np.array([[1.], [2.], [1.2]]), 2., X_train, Y_train
grads, cost = propagate(w, b, X, Y)
#print("dw: " + str(grads["dw"]))
#print("db: " + str(grads["db"]))
#print("cost: " + str(cost))

cost.shape
cost
#X_train.shape
#Y_train.shape

#np.array([[1., 2., -1.], [3., 4., -3.2], [1., -1.1, 2.], [1.,1.,1.]]).shape
# optimise()
def optimise(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    # Update parameters w and b using gradient descent
    # Arguments:
    # w - weights, a numpy array of size (num_px, 1)
    # b - bias, a scalar
    # X - data of shape (num_px, number of examples m)
    # Y - true "label" vector of shape (1, number of examples m)
    # num_iterations - number of iterations of the optimisation loop
    # learning_rate - learning rate of the gradient descent update rule
    # print_cost - True to print the loss every 100 iterations
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives
        dw = grads["dw"]
        db = grads["db"]
        
        # Update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record costs
        if i % 100 == 0:
            costs.append(cost)
            
        # Print the costs after every 100 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            
    params = {
        "w" : w,
        "b" : b
    }
    
    grads = {
        "dw" : dw,
        "db" : db
    }
        
    return params, grads, costs
# Test optimise()
params, grads, costs = optimise(w, b, X, Y, num_iterations = 1000, learning_rate = 0.009, print_cost = True)
print("w: " + str(params["w"]))
print("b: " + str(params["b"]))
print("dw: " + str(grads["dw"]))
print("db: " + str(grads["db"]))

# predict()
def predict(w, b, X):
    # Predict whether label is 0 or 1 using logistic regression via parameters w and b
    # Arguments:
    # w - weights, a numpy array of size (num_px, 1)
    # b - bias, a scalar
    # X - data, a numpy array of shape (num_px, number of examples m)
    # Returns:
    # Y_prediction - predicted labels 0 or 1, a numpy array of shape (1, number of examples m)
    m = X.shape[1]
    Y_prediction = np.zeros(shape = (1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector A to derive probabilities between 0 and 1 of image being ad or not
    A = sigmoid(np.dot(w.T,X) + b)
    
    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

# Test predict()
w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2], [1.2, 2., 0.1]])
print("predictions: " + str(predict(w, b, X)))
# model()
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.01, print_cost = False):
    # Builds the logistic regression model based on the helper functions created above
    # Arguments:
    # X_train - train set, numpy array of size (num_px, number of examples m)
    # Y_train - train label vector, numpy array of size (1, number of examples m)
    # X_test - test set, numpy array of size (num_px, number of examples m)
    # Y_test - test label vector, numpy array of size (1, number of examples m)
    # num_iterations - a scalar
    # learning_rate - a scalar
    # print_cost - True to print cost for every 100 iterations
    # Returns:
    # d - dictionary of costs, predictions, and parameters w and b, learning rate and number of iterations
    
    # Initialise parameters w and b with zeros
    w, b = initialise_with_zeros(X_train.shape[0])
    
    # Gradient descent
    parameters, grads, costs = optimise(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict train and test sets
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    # Print train and test set errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {
        "costs": costs,
        "Y_prediction_train": Y_prediction_train,
        "Y_prediction_test": Y_prediction_test,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    
    return d
d = model(X_train, Y_train, X_test, Y_test, num_iterations = 3000, learning_rate = 0.0001, print_cost = True)

# Plot learning curve with costs
costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate: " + str(d["learning_rate"]))
plt.show()
