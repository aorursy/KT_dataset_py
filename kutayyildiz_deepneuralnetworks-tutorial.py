# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def sigmoid(z, prime = None):
    sigmoid = lambda t: 1.0/(1.0 + np.exp(-t))
    sigmoid_func = np.vectorize(sigmoid)
    if not prime:
        return sigmoid_func(z)
    prime = lambda t: sigmoid_func(t) * (sigmoid_func(t) + -1)
    prime_func = np.vectorize(prime)
    return prime_func(z)
def relu(z, prime = None):
    relu = lambda t: t if t >= 0.0 else 0.0
    relu_func = np.vectorize(relu)
    if not prime:
        return relu_func(z)
    prime = lambda t: 1.0 if t >= 0.0 else 0.0
    prime_vectorized = np.vectorize(prime)
    return prime_vectorized(z)
def linear(z, prime = None):
    if not prime:
        return z
    return 1
def tanh(z, prime = None):
    if not prime:
        return np.tanh(z)
    x = 1.0 - np.tanh(z)**2
    return x
def leaky_relu(z, prime = None):
    relu = lambda t: t if t > 0.0 else 0.01 * t
    relu_func = np.vectorize(relu)
    if not prime:
        return relu_func(z)
    prime = lambda t: 1.0 if t > 0.0 else 0.01
    prime_func = np.vectorize(prime)
    return prime_func(z)
def L1(z, target = None, action = None): # Absolute error
    if not action:
        return z
    elif action == "prime":
        prime = lambda y, t: -1.0 if y < t else 1.0
        prime_vectorized = np.vectorize(prime)
        return prime_vectorized(z, target)
    elif action == "loss":
        return np.sum(np.absolute(z - target))
def L2(z, target = None, action = None): # Squared error
    if not action:
        return z
    elif action == "prime":
        return z - target
    elif action == "loss":
        return np.sum(np.square(z - target))
class Layer:
    def __init__(self, height, width, activation):
        if activation == relu or activation == leaky_relu:
            self.weight_matrix = np.random.rand(height, width) * np.sqrt(2/width)
        elif activation == tanh:
            self.weight_matrix = np.random.rand(height, width) * np.sqrt(1/width)
        else:
            self.weight_matrix = np.random.rand(height,width)
        self.bias_matrix = np.zeros((height, width))
        self.activation = activation
    def forward_feed(self, input_matrix):
        return self.activation(
            np.dot(self.weight_matrix, input_matrix)) + np.sum(self.bias_matrix, axis=1)
    def derivative(self, left_input, right_input):
        return (self.weight_matrix.T * self.activation(np.sum(right_input, axis=0), 1)).T * left_input.T
    def derivative_out(self, left_input, right_input, target_input):
        return (self.weight_matrix.T * self.activation(right_input, target_input, "prime").T).T * left_input.T
    def update_weights(self, derivative_matrix, learning_rate):
        self.weight_matrix -= learning_rate * derivative_matrix
        self.bias_matrix -= learning_rate * derivative_matrix
class Network:
    # Initialization function
    def __init__(self, input_size, output_size):
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        
    # Add Layers
    def add_out_layer(self, loss_function = L1):
        l = Layer(self.output_size, self.layers[-1].weight_matrix.shape[0], loss_function)
        self.layers.append(l)
        
    def add_layer(self, height, activation = relu):
        if not self.layers:
            width = self.input_size
        else:
            width = self.layers[-1].weight_matrix.shape[0]
        input_height = self.layers
        l = Layer(height, width, activation)
        self.layers.append(l)
    # Predict an input
    def predict(self, input_matrix, return_all = False):
        last_out = input_matrix
        out_list = [input_matrix]
        for layer in self.layers:
            last_out = layer.forward_feed(last_out)
            if return_all:
                out_list.append(last_out)
        if return_all:
            return out_list
        return last_out
    # back_prop an (input, target) pair and update weights
    def back_prop(self, input_matrix, target_matrix):
        out_list = self.predict(input_matrix, 1)
        derivative_matrix = self.layers[-1].derivative_out(out_list[-2], out_list[-1], target_matrix)
        self.layers[-1].update_weights(derivative_matrix, self.learning_rate)
        i = -3
        for layer in reversed(self.layers[:-1]):
            derivative_matrix = layer.derivative(out_list[i], derivative_matrix)
            layer.update_weights(derivative_matrix, self.learning_rate)
            i -= 1
        return self.layers[-1].activation(out_list[-1], target_matrix, "loss")
    def train(self, input_list, target_list, learning_rate = 0.1):
        batch_loss = 0.0
        self.learning_rate = learning_rate
        for input_matrix, target_matrix in zip(input_list, target_list):
            batch_loss += self.back_prop(input_matrix, target_matrix)
        return batch_loss
    def batch_train(self, input_list, target_list, learning_rate = 0.1):
        self.learning_rate = learning_rate
        loss_before = self.loss(input_list, target_list)
        best_layer_model = self.layers
        best_loss = self.layers[-1].activation(self.predict(input_list[0]), target_list[0], "loss")
        
        for input_matrix, target_matrix in zip(input_list[1:], target_list[1:]):
            current_loss = self.back_prop(input_matrix, target_matrix)
            if best_loss < current_loss:
                self.layers = best_layer_model
            else:
                best_loss = current_loss
                best_layer_model = self.layers
            return loss_before - self.loss(input_list, target_list)
    def loss(self, input_list, target_list):
        loss = 0.0
        for x, y in zip(input_list, target_list):
            loss += self.layers[-1].activation(self.predict(x), y, "loss")
        return loss
test_data = pd.read_csv("../input/testSimple.csv")
train_data = pd.read_csv("../input/trainSimple.csv")
n = Network(6, 2)
n.add_layer(24, leaky_relu)
n.add_layer(24, tanh)
n.add_out_layer(L1)
from sklearn.utils import shuffle
learning_rate = 0.001
batch_size = 12
epoch = 20
previous_total_loss = n.loss(train_data.values[:, :6], train_data.values[:, 6:])
for i in range(epoch):
    x = shuffle(train_data.values)
    improvement = 0
    for batch in np.array_split(x, len(x)/batch_size):
        improvement += n.batch_train(batch[:, :6], batch[:, 6:], learning_rate)
    epoch_loss = n.loss(train_data.values[:, :6], train_data.values[:, 6:])
    print("MAE loss for epoch:" + str(i) + " = " + str(epoch_loss))
    print("Improvement = " + str(improvement))
current_total_loss =n.loss(train_data.values[:, :6], train_data.values[:, 6:])
print("Current Total Loss = " + str(current_total_loss))
print("Total Improvement(MAE) = " + str(previous_total_loss - current_total_loss))
sample_data = shuffle(train_data)
train_inputs = sample_data.iloc[:, :6].values
train_targets = sample_data.iloc[:, 6:].values
data_number = 0
for train_input, train_target in zip(train_inputs[:10], train_targets[:10]):
    data_number += 1
    print("Sample Data " + str(data_number))
    print("Prediction:")
    prediction = n.predict(train_input)
    print(prediction)
    print("Target:")
    print(train_target)
    print("L1 loss:")
    print(L1(prediction, train_target, "loss"))
test = pd.read_csv("../input/testSimple.csv")
test.tail()
predicted_list = []
for row in test.values:
    predicted_list.append(np.append(row[0], n.predict(row[1:])))
predicted_df = pd.DataFrame(predicted_list, columns = ['ID', 'A', 'B'])
predicted_df['ID'] = predicted_df['ID'].astype(int)
predicted_df.head()
predicted_df.to_csv('submission.csv', index=False)