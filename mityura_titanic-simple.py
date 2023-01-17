import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))

def sigmoid(z):
    sig = 1 / (np.exp(-z) + 1)
    return sig
def init(dim):
    w = np.random.randn(dim) * np.sqrt(2 / dim)
    b = np.random.randn()
    return w, b
def prop(w, b, X, Y):
    m = X.shape[0]
    A = sigmoid(np.dot(w.T, X.T) + b)
    A = np.squeeze(A)
    cost = (- 1 / m) * np.sum(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))
    dw = (1 / m) * np.dot(X.T, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
def opt(w, b, X, Y, num_iterations, learning_rate):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = prop(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw 
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
        
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
def predict(w, b, X):
    m = X.shape[0]
    Y_pr = np.zeros(m)
    A = sigmoid(np.dot(w.T, X.T) + b)
    A = np.squeeze(A)
    
    for i in range(A.shape[0]):
        Y_pr[i] = 1 if A[i] > 0.5 else 0
        
    return Y_pr
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = init(X_train.shape[1])
    
    parameters, grads, costs = opt(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
train_ds = pd.read_csv('../input/train.csv')
test_ds = pd.read_csv('../input/test.csv')
X_train = train_ds.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"],axis=1)
Y_train = train_ds["Survived"]
X_test = test_ds
Y_test = np.zeros([test_ds.shape[0], 1])
m_train = Y_train.shape[0]
m_test = Y_test.shape[0]
X_train['Sex'] = X_train['Sex'].map( {'female': 2, 'male': 1} ).astype(int)
X_train['Embarked'] = X_train['Embarked'].map( {'Q' : 3, 'S' : 2, 'C' : 1} ).astype(float)
X_train = X_train.fillna(0)
X_test = test_ds.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
X_test['Sex'] = X_test['Sex'].map( {'female': 2, 'male': 1} ).astype(int)
X_test['Embarked'] = X_test['Embarked'].map( {'Q' : 3, 'S' : 2, 'C' : 1} ).astype(float)
X_test = X_test.fillna(0)
d = model(X_train, Y_train, X_test, Y_test, num_iterations = 50000, learning_rate = 0.006)
my_submission = pd.DataFrame({'Id': test_ds['PassengerId'], 'Prediction': d['Y_prediction_test']})
my_submission.to_csv('submission.csv', index=False)
