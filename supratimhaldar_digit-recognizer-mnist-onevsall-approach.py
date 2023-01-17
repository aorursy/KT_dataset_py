import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import optimize # to use minimize function

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# =============================================================================
# Function to calculate value of Sigmoid Function of any variable z.
# z can be a matrix, vector or scalar
# sigmoid g(z) = 1/(1 + e^-z)
# =============================================================================
def sigmoid(z):
    sig = 1.0/(1.0 + np.exp(-z))
    
    # Due to floating point presision related issues, e^-z might return very 
    # small or very large values, resulting in sigmoid = 1 or 0. Since we will
    # compute log of these values later in cost function, we want to avoid 
    # sig = 1 or 0, and hardcode to following values instead.
    sig[sig == 1.0] = 0.9999
    sig[sig == 0.0] = 0.0001
    
    return sig
# =============================================================================
# Compute cost of Logistic Regression with multiple features
# Vectorized implementation
# Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector
# Output: cost = 1-dim vector
# =============================================================================
def computeCost(theta, data_X, data_y, lambda_reg = 0):
    m = len(data_X) # No of rows
    n = len(data_X[0]) # No of features
    theta = theta.reshape(n,1)
    
    # h(x) = g(z) = g(theta0 + theta1*X1 + theta2*X2 + .. + thetan*Xn)
    # h(x) = g(X * theta) = Sigmoid(X * theta) = m-dim vector
    hx = sigmoid(np.dot(data_X, theta))
    cost = - np.dot(data_y.T, np.log(hx)) - np.dot((1 - data_y).T, np.log(1 - hx))
    
    # This is unregularized cost
    J = cost/m
    
    # Adding regularization. Setting theta0 to 0, because theta0 will not be 
    # regularized
    J_reg = (lambda_reg/(2*m)) * np.dot(theta[1:,:].T, theta[1:,:])
    J = J + J_reg
    
    return J
# =============================================================================
# Compute gradient or derivative of cost function over parameter, i.e.
# d J(Theta)/d Theta
# =============================================================================
def computeGradient(theta, data_X, data_y, lambda_reg = 0):
    m = len(data_X) # No of rows
    n = len(data_X[0]) # No of features
    theta = theta.reshape(n,1)
    theta_gradient = np.zeros(theta.shape)
    cost = 0
    #print("==== Inside computeGradient() ====", data_X.shape, data_y.shape)

    cost = computeCost(theta, data_X, data_y, lambda_reg)
    
    hx = sigmoid(np.dot(data_X, theta))
    error = hx - data_y
    theta_gradient = (1/m) * (np.dot(data_X.T, error))
    
    # Apply regularization
    theta_reg = (lambda_reg/m) * theta[1:,:]
    theta_gradient[1:,:] = theta_gradient[1:,:] + theta_reg
    
    #print("==== Inside computeGradient() ====", cost)
    return cost.flatten(), theta_gradient.flatten()
# =============================================================================
# One vs All method of logistic regression
# Used for data with multiple clssification outputs
# =============================================================================
def oneVsAll(data_X, data_y, num_labels, lambda_reg):
    n = data_X.shape[1] # No of features
    all_theta = np.zeros([num_labels, n])
    initial_theta = np.zeros([n, 1])
    
    for label in range(num_labels):
        theta_optimized = optimize.minimize( \
            computeGradient, \
            initial_theta, \
            args=(data_X, data_y == label, lambda_reg), \
            method = "CG", \
            jac=True, options={'disp': True, 'maxiter': 150} \
            )
        #print("OneVsAll: Optimization Result =", theta_optimized)
        theta = theta_optimized.x.reshape(n, 1)
        all_theta[label,:] = theta.T

    return all_theta
# Load training data
def loadTrainingData(path):
    train_data = pd.read_csv(path)
    #print(train_data.isnull().sum())
    return train_data
# Test my implementation of one-vs-all logistic regression algorithm
def test_OneVsAll():
    # This dataset is downloaded from Kaggle
    train_data = loadTrainingData('../input/train.csv')

    # Total number of records
    m = len(train_data)
    
    # Populate y data into a m-dim vector
    # And then drop that column from feature list
    num_labels = len(train_data.label.unique())
    data_y = train_data.label.values.reshape(m, 1)
    train_data = train_data.drop('label', 1)
    
    # Setting first feature to 1, this is the bias/y-intercept or theta0
    train_data.insert(0, 'first_dummy_feature', 1)

    # Populate X (features) data into a mxn matrix
    data_X = train_data.values
    
    # Call one-vs-all calculation
    lambda_reg = 0.5
    all_theta = oneVsAll(data_X, data_y, num_labels, lambda_reg)
    print("OneVsAll: Theta after Advanced Optimization =", all_theta.shape)
    
    # Predict results of test data
    test_data = loadTrainingData('../input/test.csv')
    test_data_m = len(test_data)
    test_data.insert(0, 'first_dummy_feature', 1)
    test_data_X = test_data.values
    
    Z = sigmoid(np.dot(test_data_X, all_theta.T))
    prediction = np.argmax(Z, axis=1)
    print("OneVsAll: Prediction Result =", prediction.shape)
    
    # Prepare submission file
    my_submission = pd.DataFrame({ \
            'ImageId': np.arange(1, test_data_m+1), \
            'Label': prediction.flatten()})
    my_submission.to_csv('DG_submission.csv', index=False)
test_OneVsAll()