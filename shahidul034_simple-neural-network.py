# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from numpy import exp, array, random, dot



class NeuralNetwork():

    def __init__(self):

        # Seed the random number generator, so it generates the same numbers

        # every time the program runs.

        random.seed(1)



        # We model a single neuron, with 3 input connections and 1 output connection.

        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1

        # and mean 0.

        self.synaptic_weights = 2 * random.random((3, 1)) - 1



    # The Sigmoid function, which describes an S shaped curve.

    # We pass the weighted sum of the inputs through this function to

    # normalise them between 0 and 1.

    def __sigmoid(self, x):

        return 1 / (1 + exp(-x))



    # The derivative of the Sigmoid function.

    # This is the gradient of the Sigmoid curve.

    # It indicates how confident we are about the existing weight.

    def __sigmoid_derivative(self, x):

        return x * (1 - x)



    # We train the neural network through a process of trial and error.

    # Adjusting the synaptic weights each time.

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

        for iteration in range(number_of_training_iterations):

            # Pass the training set through our neural network (a single neuron).

            output = self.think(training_set_inputs)

            # print("\nOutput of the Above Function After Sigmoid Applied: \n",output)



            # Calculate the error (The difference between the desired output

            # and the predicted output).

            error = training_set_outputs - output

            # print("\nTraining Set Output Matrix: \n", training_set_outputs)

            # print("\nError: Training Set Output Matrix 4x1 - Above Matrix 4x1 \n", error)



            # Multiply the error by the input and again by the gradient of the Sigmoid curve.

            # This means less confident weights are adjusted more.

            # This means inputs, which are zero, do not cause changes to the weights.

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # print("\nAdjustment Matrix: \n", adjustment)



            # Adjust the weights.

            self.synaptic_weights += adjustment



    # The neural network thinks.

    def think(self, inputs):

        dot_product = dot(inputs, self.synaptic_weights)

        # print("\nDot Product of Input Matrix and Weight Matrix: \n",dot_product)

        # Pass inputs through our neural network (our single neuron).

        return self.__sigmoid(dot_product)



if __name__ == "__main__":



    #Intialise a single neuron neural network.

    neural_network = NeuralNetwork()



    print ("\n\nRandom starting synaptic weights: ")

    print (neural_network.synaptic_weights)



    # The training set. We have 4 examples, each consisting of 3 input values

    # and 1 output value.

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

    training_set_outputs = array([[0, 1, 1, 0]]).T



    # Train the neural network using a training set.

    # Do it 10,000 times and make small adjustments each time.

    neural_network.train(training_set_inputs, training_set_outputs, 10000)



    print ("\nNew synaptic weights after training: ")

    print (neural_network.synaptic_weights)



    # Test the neural network with a new situation.

    print ("\nConsidering new situation [1, 0, 0] -&amp;amp;amp;amp;amp;amp;amp;amp;amp;gt; ?: ")

    print (neural_network.think(array([1, 0, 0])))