# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## start coding here

class Neural_Net():

    def __init__(self):

        np.random.seed(1) # make sure random is random!

        # make a [3,1] array filled with random numbers between [-1, 1] 

        self.weights = 2 * np.random.random((3,1)) - 1 

    

    # sigmoid function

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x)) # 1/(1+e^(-x))

    

    # sigmoid derivative function

    def sigmoid_derivative(self, x):

        return x * (1 - x)

    

    # train 

    def train(self, training_input, training_output, training_iteration):

        for _ in range(training_iteration):

            output = self.think(training_input)

            error = training_output - output

            # T = transpose 

            # np.dot() = dot product of two arrays

            adjustments = np.dot(training_input.T, error * self.sigmoid_derivative(output))

            self.weights += adjustments 

    

    # think

    def think(self, inputs): 

        inputs = inputs.astype(float)

        output = self.sigmoid(np.dot(inputs, self.weights))

        return output 



# Python's main function thing 

if __name__ == "__main__":

    neural_network = Neural_Net()

    

    training_input = np.array([[0,0,1],

                               [1,1,1],

                               [1,0,0],

                               [0,1,0]])

    

    training_output = np.array([[0,1,1,0]]).T

    

    #                                               iterations 

    neural_network.train(training_input, training_output, 10000)

    

    # Get input for testing 

    print("Enter 0 or 1")

    A = input("First number: ")

    B = input("Second number: ")

    C = input("Third number:")

    

    print("New input is: ", A, B, C)

    print("Predicted output is: ")

    print(neural_network.think(np.array([A, B, C])))

    

# Congrats!!!1!!!!! You've made a single node neural network :) 



# fb.com/carletonaisociety

# instagram.com/carletonaisociety