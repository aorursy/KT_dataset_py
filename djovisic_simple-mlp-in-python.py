# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy

import scipy.special

import scipy.misc

import scipy.ndimage

import matplotlib.pyplot as plt

import matplotlib.pyplot 

import datetime

%matplotlib inline
# neural network definition

class neuralNetwork:

    

    #init neural network

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):

        #set number of nodes in each input, hidden and output layer

        self.inodes = inputnodes

        self.onodes = outputnodes

        self.hnodes = hiddennodes

        

        #learning rate

        self.lr = learningrate

        

        # linkweight matrices, wih and who

        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes))

        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        

        #activation function (sigmoid function expit)

        self.activation_function = lambda x: scipy.special.expit(x)

        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        

        pass

    

    # train the neural network

    def train(self,inputs_list,targets_list):

        inputs = numpy.array(inputs_list,ndmin=2).T

        targets = numpy.array(targets_list,ndmin=2).T

        

        # calculate signals into hidden layer

        hidden_inputs = numpy.dot(self.wih,inputs)

        # calculate signals emerging from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        

        # calculate signals into final output layer

        final_inputs = numpy.dot(self.who,hidden_outputs)

        # calculate signals emerging from hidden layer

        final_outputs = self.activation_function(final_inputs)

        

        #output layer error is target - actual

        output_errors = targets - final_outputs

        #hidden layer error is the outputs error

        hidden_errors = numpy.dot(self.who.T,output_errors)

        

        #update the weights for the links between the hidden and output layers

        self.who += self.lr*numpy.dot(output_errors*final_outputs*(1.0-final_outputs),numpy.transpose(hidden_outputs))

        

        #update the weights for the links between the input and hidden layers

        self.wih += self.lr*numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs),numpy.transpose(inputs))

        

        pass

    

    #query the neural network

    def query(self,inputs_list):

        #convert inputs_list to 2d array

        inputs = numpy.array(inputs_list,ndmin=2).T

        

        # calculate signals into hidden layer

        hidden_inputs = numpy.dot(self.wih,inputs)

        # calculate signals emerging from hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        

        # calculate signals into final output layer

        final_inputs = numpy.dot(self.who,hidden_outputs)

        # calculate signals emerging from hidden layer

        final_outputs = self.activation_function(final_inputs)

        

        return final_outputs

        pass
def frange(start, stop, step):

     i = start

     while i < stop:

         yield i

         i += step

pass
rates = []

performance = []

#for nn in frange(60,200,20):

n= neuralNetwork(784,200,10,0.01)

#print("NN",nn)

# start time

starttime = datetime.datetime.now()



#load the mnist training data CSV file into a list

training_data_file = open("../input/train.csv",'r')

training_data_list = training_data_file.readlines()

training_data_file.close()



#train the neural network

epochs=10



# go through all records in the training data set

for e in range(epochs):

    for record in training_data_list[2:]:

        #split the record by ',' commas

        all_values = record.split(',')

        #scale and shift the inputs

        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

        #create the target output values (all 0.01 except the desired label which is 0.99)

        targets = numpy.zeros(10)+0.01

        #all_values[0] is the target label for this record

        targets[int(all_values[0])]=0.99

        n.train(inputs,targets)



        ## create rotated variations

        # rotated anticlockwise by x degrees

        #inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)

        #n.train(inputs_plusx_img.reshape(784), targets)

        # rotated clockwise by x degrees

        #inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)

        #n.train(inputs_minusx_img.reshape(784), targets)

    pass 



    #load the mnist test data CSV file into a list

    test_data_file = open("../input/test.csv",'r')

    test_data_list = test_data_file.readlines()

    test_data_file.close()



    #test the neural network



    #scorecard for how well the network performs, initally empty 

    pred = [[]]

    i = 1

    for record in test_data_list[2:]:

        all_values = record.split(',')

        inputs = (numpy.asfarray(all_values[0:])/255.0*0.99)+0.01

        outputs = n.query(inputs)

        label=numpy.argmax(outputs)

        pred[i,1] = i

        pred[i,2] = label

    pass





np.savetxt('submission_MLP_NN.csv', pred, delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')



# start time

endtime = datetime.datetime.now()

print("Total execution time:",endtime-starttime)

pass