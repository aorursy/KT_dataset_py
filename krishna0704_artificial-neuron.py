# Importing libraries

import numpy as np

import math
# Creating a Neuron

class Neuron(object):

    '''Creating weights and bias'''

    def __init__(self):

        self.weights = np.array([1.0, 1.5])

        self.bias = 0.0

        

    '''Creating a Forward passs'''

    def forward(self, inputs):

        z = np.sum(inputs * self.weights) + self.bias

        output = (1/(1 + math.exp(-z))) # Sigmoid Activation Function

        return output
neuron = Neuron()

predict = neuron.forward(np.array([1,2]))
print(predict)