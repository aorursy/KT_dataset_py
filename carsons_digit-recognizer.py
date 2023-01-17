# connecting packeges

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

# Sigmoid

from scipy.special import expit
class NeuralNetwork:

    # initialization function

    def __init__(self, inputnodes, hiddenodes, outputnodes, learningrate):

        # Neuron count

        ## in the incoming layer

        self.inodes = inputnodes

        ## in the hidden layer

        self.hnodes = hiddenodes

        ## in the outgoing layer

        self.onodes = outputnodes

        

        # the weight of the output from the input layer to the hidden layer

        self.wih = np.random.normal(.0, pow(self.inodes, -.5), (self.hnodes, self.inodes))

        # the weight of the output from the hidden layer to the output layer

        self.who = np.random.normal(.0, pow(self.hnodes, -.5), (self.onodes, self.hnodes)) 

        

        # Coefficient learning(the accuracy of the model)

        self.lr = learningrate

        

        # # actibation function

        self.activation_function = lambda x: expit(x)

        

    # Error receiving function 

    def query(self, inputs_list):

        # Convert to two-dimensional array

        inputs = np.array(inputs_list, ndmin=2).T

        

        # The incoming signal for the hidden layer

        hidden_inputs = np.dot(self.wih, inputs)

        # Outgoing signal for hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        

        # The incoming signal for the output layer

        final_inputs = np.dot(self.who, hidden_outputs)

        # Outgoing signal to the output of the hidden layer

        final_outputs = self.activation_function(final_inputs)

        

        return final_outputs

    

    # Training neural network Function

    def train(self, inputs_list, targets_list):

        # Getting output values

        # Convert to two-dimensional array

        inputs = np.array(inputs_list, ndmin=2).T

        

        # The incoming signal for the hidden layer

        hidden_inputs = np.dot(self.wih, inputs)

        # Outgoing signal for hidden layer

        hidden_outputs = self.activation_function(hidden_inputs)

        

        # The incoming signal for the output layer

        final_inputs = np.dot(self.who, hidden_outputs)

        # Outgoing signal to the output of the hidden layer

        final_outputs = self.activation_function(final_inputs)

        

        # Get the desired values 

        # Convert to two-dimensional array

        target = np.array(targets_list, ndmin=2).T

        

        # Calculation of the model error

        ## Error in the output layer

        outputs_errors = target - final_outputs

        ## Hidden layer error

        hidden_errors = np.dot(self.who.T, outputs_errors)

        

        

        # Update weights

        ## Between output layer hidden and 

        self.who += self.lr * np.dot((outputs_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))

        ## Between hidden and input layer

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs)) 
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
temp = train.drop(labels = ["label"],axis = 1).values.reshape(-1,28,28,1)

plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(temp[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
scaled_input = (np.array(train.iloc[1, 1:]) / 255 * .99) + .01
inputnodes = 784

hiddenodes = 100

outputnodes = 10

learningrate = .3



# Create neural network

nn = NeuralNetwork(inputnodes, hiddenodes, outputnodes, learningrate)
for pix in train.index:

    values = np.array(train.iloc[pix, 1:])

    inputs = (np.asfarray(values) / 255 * .99) + .01

    targets = np.zeros(outputnodes) + .01

    targets[int(train.iloc[pix, 1])] = .99

    nn.train(inputs, targets)
scorecard = []



for pix in test.index:

    values = np.array(test.iloc[pix, :])

    correct_label = int(sample.loc[pix, 'Label'])

    inputs = (np.asfarray(values) / 255 * .99) + .01

    outputs = nn.query(inputs)

    label = np.argmax(outputs)

    if label == correct_label:

        scorecard.append(1)

    else:

        scorecard.append(0)

#     print(correct_label, label, end='||')

scorecard_array = np.asarray(scorecard)

# scorecard_array.sum() / scorecard_array.size

ans = scorecard_array.sum() / scorecard_array.size

print('The accuracy of the predicted values of the neural network: %.2f%%' % (ans * 100))