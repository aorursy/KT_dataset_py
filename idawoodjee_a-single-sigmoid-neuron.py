# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import __future__
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
def linear(z):
    """Linear function."""
    return (z)

def linear_prime(z):
    """First derivative of the linear function."""
    return (1)
def sigmoid(z):
    """Sigmoid function."""
    return (1.0 / (1.0 + np.exp(-z)))

def sigmoid_prime(z):
    """First derivative of the sigmoid function."""
    return (sigmoid(z) * (1 - sigmoid(z)))
class SingleNeuron():
    """A single neuron class."""
    
    def __init__(self, unit = sigmoid, unit_prime = sigmoid_prime):
        """Initialize the class, i.e. pick the activation function
        for the output neuron given by `unit' and its derivative 
        `unit_prime`.
        
        `unit`: (function) activation function of the output neuron.
        
        `unit_prime`: (function derivative) derivative of the activation
        function given by `unit`.
        
        `w`: (integer) weight initialized randomly with mean 0 and 
        variance 1.
        
        `b`: (integer) bias initialized randomly with mean 0 and 
        variance 1.
        """
        
        #self.w = random.randn(1)
        #self.b = random.randn(1)
        self.w = 0
        self.b = -4.6
        self.unit = unit
        self.unit_prime = unit_prime
        
    
    def grad_desc(self, x, alpha, y = 1):
        """Implements one step of the gradient descent algorithm.
        
        `x`: (integer) input value.
        
        `y`: (integer) target value. Default is 1, which is what we want when
        the neuron "fires".
        
        `alpha`: (integer) learning rate. 
        """
        
        z0 = self.w * x + self.b  # (integer) weighted input
        h0 = self.unit(z0) # initial output before gradient descent is applied
        
        # Gradients of cost function wrt weight and bias
        grad_w = (h0 - y) * self.unit_prime(z0) * x
        grad_b = (h0 - y) * self.unit_prime(z0)
        
        # update the weight and bias and assign them new values
        new_w = self.w - alpha * grad_w
        new_b = self.b - alpha * grad_b
        
        self.w = new_w
        self.b = new_b
        
        # update the weighted input and assign an output
        z = self.w * x + self.b
        h = self.unit(z)
        
        return (h)
    
    
    def evaluate(self, x, alpha, epochs, y = 1):
        """Implements the gradient descent algorithm over the specified 
        number of epochs, evaluating the outputs and plotting the gradient
        descent progression.
        
        `epochs`: (integer) number of epochs/iterations to repeat training.
        """
        
        z0 = self.w * x + self.b 
        h0 = self.unit(z0)
        outputs = [h0] # list to contain outputs over all training epochs
        
        for j in range(epochs):
            out = self.grad_desc(x, alpha)
            outputs.append(out)
        
        plt.plot(range(epochs+1), outputs)
        plt.axhline(1, color = 'r', linestyle = '--')
        plt.xlabel('Number of epochs')
        plt.ylabel('Output value')
        plt.title('Neuron Learning Progression')
        
        return(outputs[-1])
    
sn = SingleNeuron()
# slow learning rate to demonstrate neuron saturation
# near output = 1 and output = 0
sn.evaluate(0.1, 1, 500)
