# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings 
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x_1 = np.load('../input/X.npy')
y_1 = np.load('../input/Y.npy')
img_size = 64 # our pixel size.

plt.subplot(1,2,1)   # subplot : sign zero
plt.imshow(x_1[205]) # show image at specified index number
plt.axis('off')      # don't show the axis of sign zero
plt.subplot(1,2,2)   # subplot : sign one
plt.imshow(x_1[822]) # I didn't remove the axis since we can see sizes of our image.
x = np.concatenate((x_1[204:409], x_1[822:1027]), axis=0) # concatenate two array along axis 0 (horizontally)
# Now we create an appropriate array for outputs consisting zeros & ones.
z = np.zeros(205)
o = np.ones(205)
y = np.concatenate((z,o), axis=0).reshape(x.shape[0], 1) # concatenate 'z' & 'o' and make it a 2D array
                                                         # (first it was a 1D array with shape (410, ) )
# let's see the shapes of input and output
print('x shape: {}'.format(x.shape))
print('y shape: {}'.format(y.shape))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
print(x_train.shape)
x_train_new = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test_new = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print('x_train new : {}\nx_test_new : {}'.format(x_train_new.shape, x_test_new.shape))
x_train = x_train_new.T
x_test = x_test_new.T
y_train = y_train.T
y_test = y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
def initialize_params(dimension):
    w = np.full((dimension, 1), 0.01) # first value of weights are 0.01
    b = 0                             # first value of bias is zero
    return w, b                       # return the necessary values that'll be used later in order to finish functions everytime

def sigmoid(z):
    y_head = 1/ (1+ np.exp(-z))
    return y_head
# Forward & Backward Propagation
def forward_backward_propagation(w, b, x_train, y_train):
    # Forward Propagation
    z = np.dot(w.T, x_train) + b                                     # z function consisting parameters w, b 
    y_head = sigmoid(z)                                              # get the probability through sigmoid fucntion
    loss = -(1 - y_train)*np.log(1 - y_head) - y_train*np.log(y_head)# the formula of loss function
    cost = (np.sum(loss)) / (x_train.shape[1])                       # cost function : sum of the loss function of every image
    
    # Backward Propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # derivative of cost function with respect to 'w'
    derivative_bias = np.sum(y_head-y_train) / x_train.shape[1]                 # derivative of cost function with respect to 'b'
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias} # creating a dictionary to store w & b values
    
    return cost,gradients   # return cost and gradients which will be used later


# Updating Parameters
def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):

    costs = [] # that'll be used for visualizate cost with respect to iteration count
    index = [] # same here as one row above

    for i in range(number_of_iterarion):    # do it as iteration count
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)  # return updated cost and gradients every iteration by calling function.
     
        w = w - learning_rate * gradients["derivative_weight"]              # update weights
        b = b - learning_rate * gradients["derivative_bias"]                # update bias
        if i%10 == 0:                                            # in every ten iterations
            costs.append(cost)                                   # store cost in list(costs)
            index.append(i)                                      # store iteration number in the list : index
            print ("cost at iteration {} : {}".format(i, cost))
    
    parameters = {"w": w,"b": b}  # most optimized weights & bias is stored in 'parameters' dictionary
    plt.plot(index,costs)         # plotting index vs costs
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters # return parameters which will be used later

# Predicting with Test Data
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)   # new "y_head" function full of probabilistic values of x_test data
    y_pred = np.zeros((1,x_test.shape[1])) # form a matrix full of zeros so as to change later
    for i in range(x_test.shape[1]):       # for every values of "y_head"
        if z[0,i] <= 0.5:                  # if y_head <= 0.5 which is our threshold,
            y_pred[0,i] = 0                # then predict it as sign zero.
        else:
            y_pred[0,i] = 1                # else if it's greater than 0.5, then let it be sign one.

    return y_pred                          # return the matrix of predicted values y_pred
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, iterations):
    
    dimension =  x_train.shape[0]      # set the dimension on the same value as images
    w,b = initialize_params(dimension) # return w & b

    parameters = update(w, b, x_train, y_train, learning_rate, iterations)  # forward & backward propagation : return brand new updated params
    
    y_pred_test  = predict(parameters["w"],parameters["b"], x_test)  # overfit check if it occured an overfitting situation, then the accuracy of-
    y_pred_train = predict(parameters["w"],parameters["b"], x_train) # test values would be significantly low as opposed to accuracy of train values.      

    print("train accuracy: {}".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100)) # train accuracy
    print("test accuracy:  {}".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))   # test acuracy - no overfitting or underfitting
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.02, iterations = 200) # perform logistic regression classification

# Grid Search

from sklearn.model_selection import GridSearchCV    # import Grid Search Cross Validation to find optimujm parameters
from sklearn.linear_model import LogisticRegression # import Logistic Regression
lr = LogisticRegression()  

parameters = {'C':np.logspace(-5,5,11), 'penalty':['l1', 'l2']} # create a dictionary  within Logistic Regression parameters inside
lr_cv = GridSearchCV(lr, parameters, cv = 10)                   # method, parameters of that method, count of Cross Validation.
lr_cv.fit(x_train.T, y_train.T)                                 # fit the model for our values

print('tuned hyperparameters : {}'.format(lr_cv.best_params_))  # Now we'll se best parameters among the "parameters"
print('best score: {}'.format(lr_cv.best_score_))               # Best score of the Logistic Regression with best parameters
# Logistic Regression

lr2 = LogisticRegression(C=1.0, penalty = 'l2') # Use the parameters within Logistic Regression
lr2.fit(x_train.T, y_train.T)                   # fit the model for our train values

print('score for test values: {}'.format(lr2.score(x_test.T, y_test.T))) # test score