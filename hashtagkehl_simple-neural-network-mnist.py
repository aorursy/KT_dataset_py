import numpy as np # linear algebra

# scipy.special for the sigmoid function expit()

import scipy.special

import scipy.misc

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot

from matplotlib.pyplot import imread

%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# neural network class definition

class neuralNetwork:

    

    # initialise the neural network

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # set number of nodes in each input, hidden, output layer

        self.inodes = inputnodes

        self.hnodes = hiddennodes

        self.onodes = outputnodes

        

        # link weight matrices, wih and who

        # weights inside the array are w_i_j, where link is from i to node j in the next layer

        # w11 w21

        # w12 w22 etc

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5),(self.hnodes, self.inodes))

        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5),(self.onodes, self.hnodes))

        

        # learning rate

        self.lr = learningrate

        

        # activation function is the sigmoid function

        self.activation_function = lambda x: scipy.special.expit(x)

        

        pass

    

    # train the neural network

    def train(self, inputs_list, targets_list):

        # convert inputs list to 2d array

        inputs = np.array(inputs_list, ndmin=2).T

        targets = np.array(targets_list, ndmin=2).T

        

        # calculate signals into hidden layer

        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        

        # calculate signals into final output layer

        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emeging from final output layer

        final_outputs = self.activation_function(final_inputs)

        

        # error is the (target - actual)

        output_errors = targets - final_outputs

        

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes

        hidden_errors = np.dot(self.who.T, output_errors)

        

        # update the weights for the links between the hidden output layers

        # self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        

        # update the weights for the links between the hidden and output layers 

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))                               

        

        pass

    

    # query the neural network

    def query(self, inputs_list):

        # convert inputs list to 2d array

        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer

        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from the hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into the final output layer

        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from the final output layer

        final_outputs = self.activation_function(final_inputs)

        

        return final_outputs



    
# number of input, hidden and output nodes

input_nodes = 784

hidden_nodes = 250

output_nodes = 10



# learning rare is 0.2 adjusted from 0.6 90%, 0.3 93%, 0.1 96%, and 0.2 94%

learning_rate = 0.1



# create instance of neural network

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# Train full MNIST data set provided by Kaggle

# test what we have trained

mnist_train_file = open("../input/mnist-wo-label/mnist_train.csv", 'r')

mnist_train_list = mnist_train_file.readlines()

mnist_train_file.close()
len(mnist_train_list)

# get the first full mnist record

all_values = mnist_train_list[0].split(',')

print(all_values[0])
image_array = np.asfarray(all_values[1:]).reshape((28,28))

matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# training the mnist_train_list dataset

# epochs is the number of times the training data set is used for training



epochs = 3



for record in mnist_train_list:

    # split the records by the commas

    all_values = record.split(',')

    # scale and shift the inputs

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # create the target output values 0.01 - 0.99

    targets = np.zeros(output_nodes) + 0.01

    # all_values[0] is the target label for this record

    targets[int(all_values[0])] = 0.99

    n.train(inputs, targets)

    pass

pass
# test what we have trained

mnist_test_file = open("../input/mnist-wo-label/mnist_test.csv", 'r')

mnist_test_list = mnist_test_file.readlines()

mnist_test_file.close()
print(mnist_test_list[0])
len(mnist_test_list)
# get the first test record

all_values = mnist_test_list[1].split(',')

print(all_values[0])

image_array = np.asfarray(all_values[1:]).reshape((28,28))

matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
# test the neural network



# scoreboard for how well the network performs, initially empty 

scorecard = []

# go through all the records in the test data set

for record in mnist_test_list:

    # split the record by the ',' commas

    all_values = record.split(',')

    # correct answer is the first value

    correct_label = int(all_values[0])

    print(correct_label, "correct label")

    # scale and shift the inputs

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # query the network

    outputs = n.query(inputs)

    # The index of the highest value corresponds to the label

    label = np.argmax(outputs)

    print(label, "networks answer")

    # append correct or incorrect  to list

    if (label == correct_label):

        # network's answer matches correct answer add 1 to the scorecard

        scorecard.append(1)

    else:

        # network's answer doesn't match the correct answer, add zero to the scorecard

        scorecard.append(0)

        pass

    pass

    
print(scorecard)
# calculate the performance score, the fraction of correct answers

scorecard_array = np.asarray(scorecard)

print ("Performance = ", scorecard_array.sum() / scorecard_array.size * 100, "% Accurate")
submission = ("../input/digit-recognizer/sample_submission.csv")

submission
