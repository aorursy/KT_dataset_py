# Author: Fatih Can Akıncı
# twitter/github/kaggle: akincifca
# First of all, I want to thank to Professor Murat Karakaya 
# for giving me the opportunity to dive deep into deep learning and
# motivate me to push my limits

# This is a tutorial for introduction to deep neural networks.
# In this notebook, I tried to construct the network layer by layer
# and tried to understand the underlying mathematics in deep learning
# libraries such as Keras by implementing the network with pure numpy.
# First import Numpy
import numpy as np
# Activation Functions for neural network nodes
# Sigmoid activation function
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))
  
# Derivative of sigmoid funcion
def sigmoid_derivative(x):
    return x * (1.0 - x)
# Lets design the network:
class Dense:
    
    def __init__(self, x, nodes, activation, lr=0.05): 
        self.inputs = x
        self.nodes = nodes
        self.activation = activation
        self.lr = lr
        # Initialize weights
        self.weights = np.random.normal(0.0, pow(self.inputs.shape[0], -0.5), (self.nodes, self.inputs.shape[0]))
        pass
    
    def feedforward(self):
        self.output = self.activation(np.dot(self.weights, self.inputs))    
        return self.output
    
    def backprop(self, prev_error, prev_w, activation_der):
        self.activation_der = activation_der
        self.error = np.dot(prev_w.T, prev_error)
        # application of the chain rule to find derivative of the loss function with respect to weights
        self.d_weights = np.dot((self.error * self.activation_der(self.output)), self.inputs.T)
        # print(d_weights)
        # update the weights with the derivative (slope) of the loss function
        self.weights += self.lr * self.d_weights     
        return self.d_weights

class Output:
    
    def __init__(self, x, nodes, activation, lr=0.05):
        self.inputs = x
        self.nodes = nodes
        self.activation = activation
        self.lr = lr
        # Initialize weights
        self.weights = np.random.normal(0.0, pow(self.inputs.shape[0], -0.5), (self.nodes, self.inputs.shape[0]))
        pass

    def feedforward(self):

        self.output = self.activation(np.dot(self.weights, self.inputs))
        return self.output

    def backprop(self, y, activation_der):

        self.y = y
        self.activation_der = activation_der
        # error
        self.output_error = self.y - self.output
        # application of the chain rule to find derivative of the loss function with respect to weights
        self.d_weights = np.dot((self.output_error * self.activation_der(self.output)), self.inputs.T)
        # print(d_weights)
        # update the weights with the derivative (slope) of the loss function
        self.weights += self.lr * self.d_weights
        return self.d_weights


# Step-by-Step Example of forward prop and back prop
# Lets take an input "X" and a target "y" as follows:
X = np.array([[0.2,0.1,0.3]]).T
y = np.array([[0.8,0.5,0.6]]).T
# Construct the NN
# Lets create our hidden layer that has 5 nodes
hiddenLayer = Dense(X, 5, sigmoid)
# Check our hidden layers random initialized weights
hiddenLayer.weights
# Feedforward the layer to get output values from hidden layer
hiddenLayer.feedforward()
# Lets create output layer which has also 3 nodes
outputLayer = Output(hiddenLayer.output, 3, sigmoid)
# Feedforward output layer and make predictions to find "y"
outputLayer.feedforward()
# Our target values were 0.8, 0.5 and 0.6
# Not bad but lets apply gradient descent and backprop our neural network to adjust the weights
dWo = outputLayer.backprop(y, sigmoid_derivative)
print("dWoutput:", dWo)
# backprop again and adjust the hidden layer weights
dWh = hiddenLayer.backprop(outputLayer.output_error, outputLayer.weights, sigmoid_derivative)
print("dWhidden:", dWh)
# Lets see the new weights for hidden layer
hiddenLayer.weights
# Lets make a new prediction with updated weights
hiddenLayer.feedforward()
outputLayer.feedforward()
# Training the network
for i in range (1000):
    outputLayer.backprop(y, sigmoid_derivative)
    hiddenLayer.backprop(outputLayer.output_error, outputLayer.weights, sigmoid_derivative)
    hiddenLayer.feedforward()
    prediction = outputLayer.feedforward()
print("Prediction after training:")
print(prediction)
print("Target values:")
print(y)
# Things to do
# Apply Regression dataset to this NN and adjust parameters
# Plot accuracy or error during training
