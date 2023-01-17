# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
%matplotlib inline
# Reading Dataset
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
# Diciding data into train and crossvalidation set
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(len(X), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Converting shapes for neural network
X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T
print(f"Training data input shape (n, m) --> {X_train.shape}")
print(f"Training data output shape (1, m) --> {y_train.shape}")
print(f"Test data input shape (n, m) --> {X_test.shape}")
print(f"Test data output shape (1, m) --> {y_test.shape}")
def initialize_weights(n):
    w = np.zeros((n, 1))
    b = 0
    return w, b
def forward_propagation(X, Y, w, b):
    m = len(X)
    Z = np.dot(w.T, X) + b
    A = 1/(1 + np.exp(-Z))
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    return A, cost
def backward_propagation(X, Y, A):
    m = len(X)
    dz = A - Y
    dw = (1/m) * np.dot(X, dz.T)
    db = (1/m) * np.sum(dz)
    return dw, db
def optimizer(X, Y, num_iterations = 100, learning_rate = 0.01):
    w, b = initialize_weights(X_train.shape[0])
    costs = []
    for i in range(num_iterations):
        A, cost = forward_propagation(X, Y, w, b)
        dw, db = backward_propagation(X, Y, A)
        w = w - learning_rate * dw        
        b = b - learning_rate * db
        
        costs.append(cost)
    return w, b
w, b = optimizer(X_train, y_train, num_iterations = 100, learning_rate = 0.5)
def predictions(X, Y, w, b):
    preds = np.zeros((1, X.shape[1]))
    m = len(X)
    Z = np.dot(w.T, X) + b
    A = 1/(1 + np.exp(-Z))
    for i in range(A.shape[1]):
        if A[0][i] >= 0.5:
            preds[0][i] = 1
        else:
            preds[0][i] = 0
    return preds
            
preds = predictions(X_test, y_test, w, b)
accuracy = (len(preds[preds == y_test])/len(y_test[0])) * 100
accuracy
preds
