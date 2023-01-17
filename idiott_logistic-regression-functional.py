import numpy as np

from matplotlib import pyplot as plt



def generate_data(size, k, b, noise = 0.):

    data = np.random.rand(size, 3)

    for i in range(size):

        if i <= size * (1 - noise):

            data[i, 2] = 1 if k * data[i, 0] + b > data[i, 1] else 0

        else:

            data[i, 2] = np.random.randint(0, 2)

    np.random.shuffle(data)

    return data



# demo

data = generate_data(100, -1, 1, noise=0.2)

plt.scatter(data[:, 0], data[:, 1], c = data[:, 2])
import numpy as np

import math

from matplotlib import pyplot as plt



def sigmoid(x):

    return 1 / (1 + math.e ** (-x))



def forward(X, W):

    Z = np.dot(W, X)

    A = sigmoid(Z)

    return A



def backward(W, A, Y, learning_rate):

    dW = np.sum((A - Y) * X, axis=1)

    W = W - learning_rate * dW

    return W, dW

    

def compute_loss(A, Y):

    loss = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    return loss



def evaluate(X, W, Y):

    A = forward(X, W)

    A = A > 0.5

    correct_num = np.sum(A == Y)

    accuracy = correct_num / X.shape[1]

    return accuracy

    

def plot_decision_boundary(model, X, W, Y):

    # Set min and max values and give it some padding

    x_min, x_max = X[1, :].min() - 0.1, X[1, :].max() + 0.1

    y_min, y_max = X[2, :].min() - 0.1, X[2, :].max() + 0.1

    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid

    #print(np.zeros((1, xx.ravel().shape[0])).shape, xx.ravel().shape)

    Z = model((np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]).T, W)

    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.ylabel('x2')

    plt.xlabel('x1')

    plt.scatter(X[1, :], X[2, :], c=Y.ravel(), cmap=plt.cm.Spectral)



X = data.T[0:2, :]

Y = data.T[2, :]

Y = Y.reshape((1, -1))

print(X.shape, Y.shape)

n = X.shape[0]

m = X.shape[1]

W = np.random.randn(1, n + 1)

print(W.shape)

X = np.vstack((np.ones((1, m)), X))



# train

for epoch in range(10000):

    A = forward(X, W)

    #print(compute_loss(A, Y), evaluate(X, W, Y))

    W, _ = backward(W, A, Y, 0.01)

    

plot_decision_boundary(forward, X, W, Y)