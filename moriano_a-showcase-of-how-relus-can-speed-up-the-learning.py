import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_raw = pd.read_csv("../input/mnist_train.csv")
test_raw = pd.read_csv("../input/mnist_test.csv")
train_raw.head()
train_raw.describe()
train = train_raw[(train_raw["label"] == 0) | (train_raw["label"] == 1)]
test =  train_raw[(train_raw["label"] == 0) | (train_raw["label"] == 1)]

train = train.head(1000)
test = test.head(500)
train.head()
test.head()
y_train = train["label"]
y_test = test["label"]
del train["label"]
del test["label"]
X_train = train
X_test = test
X_train /= 255
X_test /= 255
X_train.describe()
y_train = y_train.values
y_test = y_test.values
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T
m = X_train.shape[1]
def sigmoid(z):
    output = 1 / (1+np.exp(-z))
    return output

def relu(z):
    return np.maximum(z, 0)

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
import matplotlib.pylab as plt
def plot_function(function, title="sigmoid"):
    x = np.arange(-7, 7, 0.01)
    y = function(x)
    
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()
    
plot_function(sigmoid, "sigmoid")    
plot_function(tanh, "tanh")
plot_function(relu, "relu")

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu_derivative(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z

def tanh_derivative(z):
    return 1 - (np.power(tanh(z), 2))

def compute_derivative(z, function="sigmoid"):
    if function == "sigmoid":
        return sigmoid_derivative(z)
    elif function == "relu":
        return relu_derivative(z)
    elif function == "tanh":
        return tanh_derivative(z)
plot_function(sigmoid_derivative, "sigmoid derivative")
plot_function(tanh_derivative, "tanh derivative")
plot_function(relu_derivative, "relu derivative")
def initialize_params():
    W1 = np.random.randn(256, 784) * 0.01
    b1 = np.zeros((256, 1))
    W2 = np.random.randn(1, 256) * 0.01
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2
W1, b1, W2, b2 = initialize_params()
print("y_train shape", y_train.shape)
print("W1 shape", W1.shape, "X_train (A1) shape", X_train.shape,  "b1 shape", b1.shape)
A2 = np.dot(W1, X_train) + b1
print("W2 shape", W2.shape, "Z2 shape", A2.shape, "b2 shape", b2.shape)
print("----------------------------------------------------------------------")
print("We will doo W1 x X_train = Z1 ", W1.shape, "x" , X_train.shape, "=", W1.shape[0],",", X_train.shape[1])
print("\tThen we apply a activation function to Z1, getting A1... A1 shape will be the same as Z1, so ", X_train.shape)
print("Finally we will do W2 x Z1", W2.shape, "x", A2.shape, "=", W2.shape[0], ",", A2.shape[1])
print("\tThen we apply the sigmoid function to Z2, and we will get A2, which is our y_hat")
def forward_pass(W1, b1, W2, b2, X, m, activation="sigmoid"):
    Z1 = np.dot(W1, X) + b1
    if activation == "sigmoid":
        A1 = sigmoid(Z1)
    elif activation == "relu":
        A1 = relu(Z1)
    elif activation == "tanh":
        A1 = tanh(Z1)
        
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    y_hat = A2

    return y_hat, Z1, A1
def calculate_cost(y, y_hat):
    m = y.shape[1]
    log_probs = np.dot(y, np.log(y_hat.T)) + np.dot((1-y), np.log(1-y_hat.T))
    cost = (-1/m) * np.sum(log_probs)
    cost = np.squeeze(cost)
    return cost
y_hat, Z1, A1 = forward_pass(W1, b1, W2, b2, X_train, m)
cost = calculate_cost(y_train, y_hat)
print("Cost", cost)
def backward_pass(X, y, Z1, A1, W2, y_hat, activation="sigmoid"):
    m = y.shape[1]
    dZ2 = y_hat - y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * compute_derivative(Z1, function=activation)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2
def model(X, y, iterations=100, activation="sigmoid", lr=0.001):
    W1, b1, W2, b2 = initialize_params()
    costs = []
    for epoch in range(0, iterations+1):
        y_hat, Z1, A1 = forward_pass(W1, b1, W2, b2, X, m, activation=activation)
        
        
        
        dW1, db1, dW2, db2 = backward_pass(X, y, Z1, A1, W2, y_hat, activation=activation)
        
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
        
        if epoch % 500 == 0:
            current_cost = calculate_cost(y, y_hat)
            costs.append(current_cost)
            print("Cost at epoch", epoch, "is", current_cost)
    return costs
sigmoid_costs = model(X_test, y_test, iterations = 6000, activation="sigmoid")
tanh_costs = model(X_test, y_test, iterations = 6000, activation="tanh")
relu_costs = model(X_test, y_test, iterations = 6000, activation="relu")
x = np.arange(0, 6001, 500)
plt.figure(figsize=(12, 7))
plt.plot(x, sigmoid_costs)
plt.plot(x, tanh_costs)
plt.plot(x, relu_costs)
plt.legend([ "Sigmoid cost", "Tanh cost", "ReLU cost"])
plt.show()
