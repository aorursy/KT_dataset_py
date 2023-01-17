# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# scipy.special for sigmoid function

import scipy.special



# visualizations

import matplotlib.pyplot

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# file imports

df_mnist_test = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')

df_mnist_train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')





# load the mnist train data CSV into a list

training_data_file = open('../input/mnist-in-csv/mnist_train.csv','r')

training_data_list = training_data_file.readlines()

training_data_file.close()





# load the mnist test data CSV file into a list

test_data_file = open("../input/mnist-in-csv/mnist_test.csv", 'r')

test_data_list = test_data_file.readlines()

test_data_file.close()

df_mnist_test
print("Length of the train set: " + str(len(df_mnist_train)))

print("Length of the test set: " + str(len(df_mnist_test)))

all_values[:]
# split the CSV-file values

all_values = training_data_list[1].split(',')



# reshape the comma seperated values into an 28 x 28 array

image_array = np.asfarray(all_values[1:]).reshape((28,28))



print(image_array)



# visualize this 28x28 reshaped array

matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# scaling of the initial array values to reduce zero calculations inside the NN.

scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

print(scaled_input)
# neural network class definition

class neuralNetwork:

    

    # initialise the neural network

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # set number of nodes in each layer

        self.inodes = inputnodes

        self.hnodes = hiddennodes

        self.onodes = outputnodes



        # link weight matrices, 

        #      wih and who

        

        # 1: mean value of the normal distribution - 0.0

        # 2: standard deviation - based on the root of nodes of the upcomming layer ->

        #     pow(self.hnodes, -0.5) --- exponent -0.5 is equal to root of 

        # 3: last param builds the grid of the array (self.hnodes, self.inodes)

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))



        # learning rate 

        self.lr = learningrate



        # activation function - sigmoid function

        self.activation_function = lambda x: scipy.special.expit(x)

        

        pass

    

    #train the neural network

    def train(self, inputs_list, targets_list):

        # convert inputs list to 2d array

        inputs = np.array(inputs_list, ndmin=2).T

        targets = np.array(targets_list, ndmin=2).T

        

        # calculate signals into hindden layer

        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        

        # calculate signals into final output layer

        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging form final output layer

        final_outputs = self.activation_function(final_inputs)

        

        

        # BACKPROPAGATION #

        

        # error is the (target - actual)

        output_errors = targets - final_outputs

        

        # hidden layer error is the output_error, split by weights, recombined at hidden nodes

        hidden_errors = np.dot(self.who.T, output_errors) 

        

        # update the weights for the links between the hidden and output layers

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        

        # update the weights for the links between the input and hidden layers

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        

        pass

    

    #query the neural network

    def query(self, inputs_list):

        # convert input list to 2d array

        inputs = np.array(inputs_list, ndmin=2).T

        

        # calcuclate signals into hidden layer

        hidden_inputs = np.dot(self.wih, inputs)

        # calculate signals emerging from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        

        # calculate signals  into final output layer

        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate signals emerging from final output layer

        final_outputs = self.activation_function(final_inputs)

        

        return final_outputs

        
# number of nodes

input_nodes = 784

hidden_nodes = 100

output_nodes = 10



# learning rate with 0.1

learning_rate = 0.1



# create an instance of neuralnetwork

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# train the neural network



# epochs -> the number of times the training data set is used for training

epochs = 5



for e in range(epochs):

    # go through all records in the training data set

    for record in training_data_list[1:]:



        # split the record by the ',' commas

        all_values = record.split(',')



        # scale and shift the inputs

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01



        # create the target output values, 

        # -> an array out of output_nodes elements (all will receive the values 0.01, ...

        targets = np.zeros(output_nodes) + 0.01



        # ... except the desired label which will be set to 0.99 here) 

        # -> all_values[0] is the target label for this record.

        targets[int(all_values[0])] = 0.99



        n.train(inputs, targets)



        pass

    pass

# What is Inside the Training Elements from the Training Code up there?!

# This part here is just for visualizing the values insides the single elements

# of the calculation up there, to get a better understanding what is happening in the code.

print ( "output_nodes                   ->    ", output_nodes)  

print ( "np.zeros(output_nodes) + 0.01  ->    ", np.zeros(output_nodes) + 0.01)  

print ( "targets[int(all_values[0])]    ->    ", targets[int(all_values[0])])

print ( "int(all_values[0])             ->    ", int(all_values[0]) )  

# get the first test record

all_values = test_data_list[1].split(',')

print("Testrecord for handwritten number: " + str(all_values[0]))
image_array = np.asfarray(all_values[1:]).reshape((28,28))



matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation = 'None')
n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
# test the neural network 



# scorecard for how wel the network performs, initially empty

scorecard = []



# go through all the records in the test data set

for record in test_data_list[1:]:

    # split the record by the ',' commas

    all_values =  record.split(',')

    

    # correct answer is first value

    correct_label = int(all_values[0])

    # print(correct_label, "correct label")

    

    # scale and shift the inputs

    inputs =  (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    

    # query the network

    outputs = n.query(inputs)

    

    # the index of the highest value corresponds to the label

    label = np.argmax(outputs)

    # print(label, "network's answer")

     

    # append correct or incorrect to list

    if (label == correct_label):

        

        # network's answer matches correct answer, add 1 to scorecard 

        scorecard.append(1)

    

    else:

        # network's answer does not match correct answer, add 0 to scorecard

        scorecard.append(0)

    

    pass
# calculate the performance score, the fraction of correct answers 

scorecard_array = np.asfarray(scorecard)

print("performance = ", scorecard_array.sum() / scorecard_array.size)
import scipy.misc

import imageio

import glob
own_handwritten_digits = []





for image_file_name in glob.glob('../input/own-handwritten-digits/*.png'):

    label = int(image_file_name[-5:-4])

        

    # read picture file

    img_array =  imageio.imread(image_file_name, as_gray = True)

    

    # print(label)

    

    # the mnist-dataset pictures are stored in a contrariwise greyscale way which means 0 is white and 255 is black

    # and not as usual 0 is black and 255 is white. This leads us to the subtraction of 255 -> we make the uploaded picture equal to the rest of mnist-digits

    img_data = 255.0 - img_array.reshape(784)

       

    # rescaling the image pixels betwenn 0.01 and 1.0

    img_data = (img_data / 255.0 * 0.99) + 0.01



    record = np.append(label, img_data)

    # print(record)

    own_handwritten_digits.append(record)

    pass
# visualizing handwritten digit

matplotlib.pyplot.imshow(own_handwritten_digits[0][1:].reshape(28,28), cmap='Greys', interpolation = 'None')
# result

np.argmax(n.query(own_handwritten_digits[0][1:]))
# visualizing handwritten digit

matplotlib.pyplot.imshow(own_handwritten_digits[1][1:].reshape(28,28), cmap='Greys', interpolation = 'None')
# result

np.argmax(n.query(own_handwritten_digits[1][1:]))
# visualizing handwritten digit

matplotlib.pyplot.imshow(own_handwritten_digits[3][1:].reshape(28,28), cmap='Greys', interpolation = 'None')
# result

np.argmax(n.query(own_handwritten_digits[3][1:]))
# visualizing first handwritten digit

matplotlib.pyplot.imshow(own_handwritten_digits[2][1:].reshape(28,28), cmap='Greys', interpolation = 'None')
# result

np.argmax(n.query(own_handwritten_digits[2][1:]))