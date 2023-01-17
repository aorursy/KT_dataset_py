# set-on autocomplete

%config IPCompleter.greedy=True



import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import random



N_CLASSES = 10
data = np.load('../input/train.npz')

X_0, y_0 = np.array(data['xs'], dtype='float'), np.array(data['ys'], dtype='float')

print('x = (', type(X_0), X_0.shape, ')    y = (', type(y_0), y_0.shape, ')')
# Convert to greyscale

# Y = 0.2125 R + 0.7154 G + 0.0721 B



X = X_0[:, 0:1024] * 0.2125      # R

X += X_0[:, 1024:2048] * 0.7154  # G

X += X_0[:, 2048:3072] * 0.0721  # B



X, X.shape
y_0, y_0.shape
data_tuples = list(zip(X, y_0))

print(data_tuples)
data_tuples[100:120]
test_data = data_tuples[100:120]

training_data = data_tuples[0:100]
test_data
a = [1,2,3,4,5]

a[:-1], a[1:]
class Network(object):



    def __init__(self, sizes):

        # Sizes must be an array whose ith element represents the number of neurons on the ith layer.

        self.sizes = sizes

        self.num_layers = len(sizes)

        

        # The biases unit for each neuron of hidden and output layers.

        # The ith line represents the ith layer, and jth column, the jth cell.

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        

        # The weights for each connection between neurons.

        # The ith matrix represents the connection between the ith and (i+1)-ith layer,

        # the jth line is the jth cell on next (i+1) layer and zth the zth cell on the current (i) layer.

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        

    

    def feedforward(self, a):

        # For the given input 'a', multiply the values for the weigths,

        # add the bias units and apply the sigmoid.

        # In other words, run the input though the network.

        for bias, weight in zip(self.biases, self.weights):

            a = sigmoid(np.dot(weight, a)+bias)

        return a

    

    

    def mb_gradient_descent(self, training_data, epochs, mini_batch_size, alpha, validation_data):

        # Performs a Mini-Batch Gradient Descent

        n = len(training_data)

        for j in range(epochs):

            # Shuffle the data so the order doesn't influence

            random.shuffle(training_data)

            # Separate the mini batches 

            for k in range(0, n, mini_batch_size):

                mini_batches = [training_data[k:k+mini_batch_size]]

            # Run the optimization for each mini-batch

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, alpha)

            

            print("Epoch ", j , ": ", self.evaluate(validation_data), " / ", len(validation_data))

                

    

    def update_mini_batch(self, mini_batch, alpha):

        # Start the gradient matrixes with zeros and the same shape as the

        # original weight and bias ones.

        grad_bias = [np.zeros(bias.shape) for bias in self.biases]

        grad_weight = [np.zeros(weight.shape) for weight in self.weights]

        

        # For each pair data and class

        for x, y in mini_batch:

            # Calculate the variation with backpropagation

            delta_grad_bias, delta_grad_weight = self.backprop(x, y)

            

            # Update the gradient values with the deltas from backpropagation

            for i in range(len(grad_bias)):

                grad_bias[i] = grad_bias[i] + delta_grad_bias[i]

            for i in range(len(grad_w)):

                grad_weight[i] = grad_weight[i] + delta_grad_weight[i]

        

        # Update the weights based on the previously updted gradient values.

        for i in range(len(self.weights)):

            self.weights[i] = self.weights[i] - (alpha/len(mini_batch)) * grad_weight[i]

        for i in range(len(self.biases)):

            self.biases[i] = self.biases[i] - (alpha/len(mini_batch)) * grad_bias[i]

            

    

    def backprop(self, x, y):

        # Start the gradient matrixes with zeros and the same shape as the

        # original weight and bias ones.

        grad_bias = [np.zeros(bias.shape) for bias in self.biases]

        grad_weight = [np.zeros(weight.shape) for weight in self.weights]

        

        # Run the sample though the network (feed forward),

        # saving the outputs for each layer.

        activation = x             # Thes first value is the input.

        zs = []                    # Store all the z vectors (sum(Wi*activation_(i-1))), layer by layer

        activations = [activation] # Store all the activations (sigmoid(z)), layer by layer

        

        for bias, weight in zip(self.biases, self.weights):

            z = np.dot(weight, activation)+bias

            zs.append(z)

            activation = sigmoid(z)

            activations.append(activation)

        

        # backward pass

        # Calculate the delta and gradients for the last layer (output)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        grad_bias[-1] = delta

        grad_weight[-1] = np.dot(delta, activations[-2].transpose())

        

        # Note that, as the indexes are accessed with -l,

        # this loop will operate backwards in the arrays,

        # starting from the last-but-one element (-2).

        for i in range(2, self.num_layers):

            layer = -i

            z = zs[layer]

            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[layer+1].transpose(), delta) * sp

            grad_bias[layer] = delta

            grad_weight[layer] = np.dot(delta, activations[layer-1].transpose())

        return (grad_bias, grad_weight)

    

    

    def evaluate(self, test_data):

        # Runs the network with current weights and biases

        # for the dataset in test_data and returns the count

        # of correct predictions.

        test_results = []

        for (x, y) in test_data:

            test_results.append((np.argmax(self.feedforward(x)), y))

        return sum(int(x == y) for (x, y) in test_results)



    

    def cost_derivative(self, output_activations, y):

        # Return the vector of difference between the outputs and correct answers.

        return (output_activations-y)

    

    
def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def sigmoid_prime(z):

    """Derivative of the sigmoid function."""

    return sigmoid(z)*(1-sigmoid(z))
# net = Network([2, 3, 1])



# net.biases



# for w in net.weights:

#     print(w.shape)

    

# for b, w in zip(net.biases, net.weights):

#     print(b,'\n--------\n', w)

#     print('\n\n')
net = Network([1024, 1024, 10])
# def mb_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):

net.mb_gradient_descent(training_data, 3, 64, 0.15, test_data=test_data)
net.weights
n_test = len(test_data)

print("{} / {}".format(net.evaluate(test_data), n_test))