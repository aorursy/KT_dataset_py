



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy

import scipy

import scipy.special

#from scipy.special import softmax



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



class data_entry:

    def __init__(row):

        

        pass





def format_data(data):

    d = []

    for row in data:

        newRow = []

        

        row = row.split(',')

        row = row[1:] + [row[0]]

        for item in row:

            newRow.append(float(item))

            

        d.append(newRow)

        

        

    

    return d



#data = format_data([(1,1,1), (1,0,1), (0,1,1), (0,0,-1)]) # Or.

#data = format_data([(1,1,-1),(1,0,1),(0,1,1),(0,0,-1)]) # Xor. <- most difficult.

#data = format_data([(1,1,1),(1,0,-1),(0,1,-1),(0,0,-1)]) # And.

#data = format_data([(1,1,1),(1,0,-1),(0,1,1),(0,0,1)]) # Equivalence.



def adapt_label(label):

    return label / 10.1



def reconstruct_label(value):

    return value * 10.0



#print(data)
#for d in data:

#    print("set:", d)
class neuralNetwork:

    def __init__(self, input_example: list, n_hidden, n_output: int, learningrate: float):

        self.inodes = len(input_example)

        self.hnodes = n_hidden

        

        self.onodes = n_output

        

        # left -> right.

        # rows -> columns.

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))

        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        

        # learning rate

        self.lr = learningrate

        

        # activation function is the sigmoid function

        self.activation_function = lambda x: scipy.special.expit(x)

        #self.activation_function = lambda x: scipy.special.softmax(x)

        

        pass

    

    def train(self, inputs_list: list, targets_list: list):

        inputs = numpy.array(inputs_list, ndmin=2).T

        targets = numpy.array(targets_list, ndmin=2).T

        

        # Forward-propagate.

        sum_hidden = numpy.dot(self.wih, inputs_list)

        x_hidden = self.activation_function(sum_hidden).reshape((self.hnodes, 1))

        

        sum_out = numpy.dot(self.who, x_hidden)

        

        x_out = self.activation_function(sum_out)



        #print("x_out ", x_out.shape)

        x_out = x_out.reshape((self.onodes, 1))

        #print("x_out ", x_out.shape)

        

        #print("targets", targets.shape)

        

        # Backpropagate

        ## Errors of output.

        err_out = targets - x_out

        

        # Jeweiligen error aufteilen auf Verknüpfungen.

        err_hidden = numpy.dot(self.who.T, err_out)

        

        #print("err_out ", err_out.shape)

        #print("x_out", x_out.shape)

        #print("x_hidden", x_hidden.shape)

        

        #print("err_out * x_out", (err_out * x_out).shape)

        

        #print("self.who.shape", self.who.shape)

        #print("self.who.shape transposed", numpy.transpose(self.who).shape)

        #print("self.who", self.who)

        

        # update the weights for the links between the hidden and output layers

        self.who += self.lr * numpy.dot((err_out * x_out * (1.0 - x_out)), numpy.transpose(x_hidden))

        

        # update the weights for the links between the input and hidden layers

        self.wih += self.lr * numpy.dot((err_hidden * x_hidden * (1.0 - x_hidden)), numpy.transpose(inputs))

        pass

    

    def query(self, input):

        sum_hidden = numpy.dot(self.wih, input)

        x_hidden = self.activation_function(sum_hidden)

        

        sum_out = numpy.dot(self.who, x_hidden)

        x_out = self.activation_function(sum_out)

        

        return x_out 

# load the mnist training data CSV file into a list

training_data_file = open("../input/train.csv", 'r')

training_data_list = training_data_file.readlines()

training_data_list = training_data_list[1:]

training_data_file.close()



#data = training_data_list



data = format_data(training_data_list)



example_input = data[0][:-1]



learningrate = 0.1

output_nodes = 10



print("example input", example_input)



# Always same random numbers.

numpy.random.seed(1)



n = neuralNetwork(example_input, 50, output_nodes, learningrate)



print("wih:", n.wih)

print("who:", n.who)



epochs = 5



# Train model

for epoch in range(epochs):

    print("epoch:", epoch)

    for row in data:

        # Get inputs between 0.01 and 249.99.

        inputs = (numpy.asfarray(row[:-1]) / 255.0 * 0.99) + 0.01

        

        # Have target between 0.01 and 0.99.

        targets = numpy.zeros(output_nodes) + 0.01

        

        # Set expected label to 0.99.

        targets[int(row[-1])] = 0.99



        n.train(inputs, targets)





print("wih:", n.wih)

print("who:", n.who)
example_output = [data[0][-1]]

example_input = data[0][:-1]



print(example_output)

print(example_input)
# load the mnist test data CSV file into a list

test_data_file = open("../input/train.csv", 'r')

test_data_list = test_data_file.readlines()

test_data_list = test_data_list[1:]

test_data_file.close()



test_data_list = format_data(test_data_list)
# test the neural network



# scorecard for how well the network performs, initially empty

scoreboard = {"good": 0, "bad": 0}



# go through all the records in the test data set

for record in test_data_list:

    # correct answer is first value

    correct_label = int(record[-1])

    # scale and shift the inputs

    inputs = (numpy.asfarray(record[0:-1]) / 255.0 * 0.99) + 0.01

    # query the network

    outputs = n.query(inputs)

    # the index of the highest value corresponds to the label

    label = numpy.argmax(outputs)

    

    # append correct or incorrect to list

    if (label == correct_label):

        # network's answer matches correct answer, add 1 to scorecard

        scoreboard["good"] += 1

    else:

        # network's answer doesn't match correct answer, add 0 to scorecard

        scoreboard["bad"] += 1

    pass



# calculate the performance score, the fraction of correct answers

#scorecard_array = numpy.asarray(scorecard)

#print ("performance = ", scorecard_array.sum() / scorecard_array.size)



print('score - {:f}%'.format(scoreboard["good"] * 1.0 / len(data) * 100.0))
# load the mnist test data CSV file into a list

test_data_file = open("../input/test.csv", 'r')

test_data_list2 = test_data_file.readlines()

test_data_list2 = test_data_list2[1:]

test_data_file.close()
results = {}



import csv

with open('results.csv', 'wt') as csvfile:

    csvwriter = csv.writer(csvfile, delimiter=',')

    

    csvwriter.writerow(['ImageId', 'Label'])

    

    # go through all the records in the test data set

    for i in range (len(test_data_list2)):

        record = test_data_list2[i]

        # split the record by the ',' commas

        all_values = record.split(',')

        # scale and shift the inputs

        inputs = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01

        # query the network

        outputs = n.query(inputs)

        # the index of the highest value corresponds to the label

        label = numpy.argmax(outputs)

        # append correct or incorrect to list



        csvwriter.writerow([i + 1, label])

    

    pass



# load the mnist test data CSV file into a list

file = open("results.csv", 'r')

file_list = file.readlines()

file_list = file_list[0:]

file.close()



for row in file_list:

    print(row)