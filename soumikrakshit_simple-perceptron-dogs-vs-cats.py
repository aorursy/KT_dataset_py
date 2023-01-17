import numpy as np
import pandas as pd
import os, cv2, random, warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
print(os.listdir("../input"))
x_train, y_train = [], []
for i in range(500):
    try:
        img = cv2.imread('../input/train/cats/cat.' + str(i) + '.jpg')
        img = cv2.resize(img, (64, 64))
        x_train.append(img)
        y_train.append(0)
    except Exception as e:
        pass
for i in range(500):
    try:
        img = cv2.imread('../input/train/dogs/dog.' + str(i) + '.jpg')
        img = cv2.resize(img, (64, 64))
        x_train.append(img)
        y_train.append(1)
    except Exception as e:
        pass
x_train = np.array(x_train)
x_train.shape
y_train = np.array(y_train)
y_train = y_train.reshape(-1, 1)
x_train, y_train = shuffle(x_train, y_train, random_state = 0)
m_train = y_train.shape[1]
num_px = x_train.shape[1]
print("Number of training examples: m_train = " + str(m_train))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(x_train.shape))
print("train_set_y shape: " + str(y_train.shape))
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
x_test_flatten = x_test.reshape(x_test.shape[0], -1).T
x_train_flatten.shape, x_test_flatten.shape
x_train = x_train_flatten / 255
x_test = x_test_flatten / 255
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1  / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    grads = {"dw": dw, "db": db}
    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
#     w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):    
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
d = model(x_train, y_train, x_test, y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
