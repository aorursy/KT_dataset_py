import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
random.seed(42)

samples = 10

X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-2, 4, 8,  9,  9, 15, 16, 19, 25, 29])
w = 3
print("X", X)
print("y", y)
print("w", w)
plt.plot(X, y)
def predict(my_X, my_W, my_B):
    return np.dot(my_W, my_X) + my_B
b = 0
y_hat = predict(X, w, b)
print("y_hat\t", y_hat)
print("y\t", y)
n = samples

def error(y, y_hat):
    diff = sum(y - y_hat)
    squared_diff = diff ** 2
    error = (1/n) * squared_diff
    return error
print("Error is", error(y, y_hat))
y_hat
def derivative(X, w, b, y):
    n = len(y)
    y_hat = predict(X, w, b)
    diff_sum = sum(y-y_hat)
    
    w_derivative = (2/n) * sum((y_hat - y) * X)
    b_derivative = (2/n) * sum(y_hat-y)
    
    return w_derivative, b_derivative
w_deriv, b_deriv = derivative(X, w, b, y)
print("W_derivative", w_deriv)
print("b_derivative", b_deriv)
lr = 0.01
for iteration in range(0, 100):
    y_hat = predict(X, w, b)
    
    if iteration % 10 == 0:
        print("Iteration ", iteration, "Error", error(y, y_hat))

    W_derivative, b_derivative = derivative(X, w, b, y)
    
    w = w - (lr * W_derivative)
    b = b - (lr * b_derivative)

