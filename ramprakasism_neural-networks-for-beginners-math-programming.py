from IPython.display import Image

import os

Image("../input/neural_network_diagram.jpg")
import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn.metrics import confusion_matrix
class TwoLayerNeuralNetwork(object):

    """

    A two-layer feedforward neural network.

    arguments: data_matrix, target_variable, hidden, iteration_limit, learning_rate

    """

    def __init__(self, data_matrix = None, target_variable = None, hidden = None,

                 iteration_limit = None, learning_rate = None):

        # Create design matrix

        self.N = data_matrix.shape[0]  # assumes df is np array

        design_matrix = pd.DataFrame(data_matrix)

        design_matrix.insert(0, 'bias', np.ones(self.N))  # add column of 1's

        self.X = design_matrix

        

        # Helper variables

        D = data_matrix.shape[1]  # number of dimensions of data matrix

        self.T = pd.get_dummies(target_variable)  # one-hot encoded matrix

        K = len(self.T.columns)  # number of classes

        M = hidden  # number of chosen hidden layers

        self.Y = target_variable  # target variable

        self.tau_max = iteration_limit

        self.eta = learning_rate

        

        # Initialize random weight matrices

        # Reference: https://stackoverflow.com/questions/53988469/how-to-fix-typeerror-object-arrays-are-not-currently-supported-error-in-numpy

        # NOTE: Apparently, there are issues with object arrays and so

        # the matrices are converted to pandas data frames

        self.W_1 = pd.DataFrame(0.01 * np.random.random(((D + 1), M)))

        self.W_2 = pd.DataFrame(0.01 * np.random.random(((M + 1), K)))

    

    def tanh(self, x):

        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))



    def softmax(self, x):

        # Reference: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

        # NOTE: Used solution by PabTorre

        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    

    def forward_propogation(self):

        self.A_1 = np.dot(self.X, self.W_1)

        self.H = self.tanh(self.A_1)

        self.Z_1 = np.insert(self.H, 0, 1, axis=1)  # add column of 1's for bias

        self.A_2 = np.dot(self.Z_1, self.W_2)

        self.Yhat = self.softmax(self.A_2)

        return self

    

    def loss_function(self):

        # calculate the loss for a current iteration

        return sum(np.sum(self.T * np.log(self.Yhat), axis=1))

    

    def accuracy(self):

        # check the prediction accuracy for a current iteration

        return sum(pd.DataFrame(self.Yhat).idxmax(axis=1) == self.Y) / self.N



    def backprop(self):

        # calculate gradient for W_2

        self.delta_k = self.Yhat - self.T  # a notation borrowed from Bishop's text (Yhat - T)

        self.W_2_gradient = np.dot(np.transpose(self.Z_1), self.delta_k)

       

        # calculate gradient for W_1

        self.H_prime = 1 - self.H**2  # derivative of tanh

        self.W_2_reduced = self.W_2.iloc[1:,]  # drop first row from W_2

        self.W_1_gradient = np.dot(np.transpose(self.X),

                                   (self.H_prime * np.dot(self.delta_k,

                                                          np.transpose(self.W_2_reduced))))



        # update weights with gradient descent

        self.W_1 = self.W_1 - self.eta * self.W_1_gradient

        self.W_2 = self.W_2 - self.eta * self.W_2_gradient

        return self



    def train(self):

        for i in range(int(self.tau_max)):

            self.forward_propogation()

            self.backprop()

            if (i % 1000) == 0:

                print('Iteration: ', i,

                      'Loss: ', round(self.loss_function(), 4),

                      'Accuracy: ', round(self.accuracy(), 4))

        
iris = datasets.load_iris()  # load Iris dataset for classification

ANN = TwoLayerNeuralNetwork(data_matrix=iris.data, target_variable=iris.target,

                        hidden=5, iteration_limit=5e4, learning_rate=0.0001)  # create ANN
ANN.train()
confusion_matrix(iris.target, pd.DataFrame(ANN.Yhat).idxmax(axis=1))