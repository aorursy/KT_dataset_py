import random

import numpy as np

import math

import matplotlib.pyplot as plt


def initialize_network(n_inputs, n_hidden, n_outputs):

	network = list()

	hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]

	network.append(hidden_layer)

	output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]

	network.append(output_layer)

	return network


def sigmoid(x):

    return(1/(1+np.exp(-x)))



def tanh(x):

	return np.tanh(x)





def activate(weights, inputs):

	activation = weights[-1] #bias

	for i in range(len(weights)-1):

		activation += weights[i] * inputs[i]

	return activation
def forward_propagate(network, row,transformation):

	inputs = row

	for layer in network:

		new_inputs = []

		for neuron in layer:

			activation = activate(neuron['weights'], inputs)

			if(transformation == 'tanh'):

				neuron['output'] = tanh(activation)

			elif(transformation == 'sigmoid'):

				neuron['output'] = sigmoid(activation)

			new_inputs.append(neuron['output'])

		inputs = new_inputs

	return inputs
def sigmoid_backward_propagation(x):

    return x * (1.0 - x)



def tanh_deriv(x):

    return 1.0 - np.tanh(x)**2
def backward_propagate_error(network, expected,transformation ):

	for i in reversed(range(len(network))):

		layer = network[i]

		errors = list()

		if i != len(network)-1:

			for j in range(len(layer)):

				error = 0.0

				for neuron in network[i + 1]:

					error += (neuron['weights'][j] * neuron['delta'])

				errors.append(error)

		else:

			for j in range(len(layer)):

				neuron = layer[j]

				errors.append(expected - neuron['output'])

		for j in range(len(layer)):

			neuron = layer[j]

			if(transformation == 'tanh'):

				neuron['delta'] = errors[j] * tanh_deriv(neuron['output'])

			elif (transformation == 'sigmoid'):

				neuron['delta'] = errors[j] * sigmoid_backward_propagation(neuron['output'])
def update_weights(network, row, l_rate):

	for i in range(len(network)):

		inputs = row[:-1]

		if i != 0:

			inputs = [neuron['output'] for neuron in network[i - 1]]

		for neuron in network[i]:

			for j in range(len(inputs)):

				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]

			neuron['weights'][-1] += l_rate * neuron['delta']
def train_network(network, train, l_rate, n_epoch, n_outputs,transformation):

	for epoch in range(n_epoch):

		sum_error = 0

		for row in train:

			outputs = forward_propagate(network, row,transformation)

			expected = row[-1]

			sum_error += ((expected-outputs[0])**2 )

			backward_propagate_error(network, expected,transformation)

			update_weights(network, row, l_rate)

		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
def predict(network, row,transformation):

	outputs = forward_propagate(network, row,transformation)

	return outputs[0]
def func(evalu,x):

   if(evalu == "x*x"):

      return x * x

   elif (evalu == "sin"):

      return np.sin(x)

   elif (evalu == "cos"):

      return np.cos(x)

   elif (evalu == "exp"):

      return np.exp(x)

   elif (evalu == "x"):

      return x 
EVALUATION_FUNCTION = 'x'

# ACTIVATION_FUNCTION = 'sigmoid'

ACTIVATION_FUNCTION = 'tanh'

RANGE_START = 0

RANGE_END = 100

RANGE_COUNT = 1000

LEARNING_RATE = .1

NUMBER_OF_EPOCHS = 400

def normalize(x):

    """

        argument

            - x: input image data in numpy array [32, 32, 3]

        return

            - normalized x 

    """

    min_val = np.min(x)

    max_val = np.max(x)

    x = (x-min_val) / (max_val-min_val)

    return x
random.seed(1)



originalDataInput = np.random.uniform(low=RANGE_START, high=RANGE_END, size=(RANGE_COUNT,))

triFunc = ['sin','cos','tan']

if EVALUATION_FUNCTION in triFunc:

    originalDataInput = originalDataInput * math.pi / 180

originalDataOutput = func(EVALUATION_FUNCTION,originalDataInput)



normalizedDataInput = normalize(originalDataInput)

normalizedDataOutput = normalize(originalDataOutput)

dataset = np.column_stack([normalizedDataInput, normalizedDataOutput])
n_inputs = 1

n_outputs = 1

network = initialize_network(n_inputs, 5, n_outputs)
train_network(network, dataset, LEARNING_RATE, NUMBER_OF_EPOCHS, n_outputs,ACTIVATION_FUNCTION)
originalDataInput = np.random.uniform(low=0, high=100, size=(10,))

triFunc = ['sin','cos','tan']

if EVALUATION_FUNCTION in triFunc:

    originalDataInput = originalDataInput * math.pi / 180

originalDataOutput = func(EVALUATION_FUNCTION,originalDataInput)



normalizedDataInput = normalize(originalDataInput)

normalizedDataOutput = normalize(originalDataOutput)

dataset = np.column_stack([normalizedDataInput, normalizedDataOutput])
sumError = 0

count = 0

predictedOutput = []

for row in dataset:

   prediction = predict(network, row,ACTIVATION_FUNCTION)

   count+=1

   error = math.pow(row[-1]-prediction,2)

   sumError += error

   expected = row[-1]

   row[-1] = prediction 

   print('Input=%f ,Expected=%f, Got=%f, error=%f' % (row[0],expected, prediction, error))

plt.plot(normalizedDataInput,normalizedDataOutput,'o',label='Actual')

plt.plot(dataset[:,0],dataset[:,1],'o',label='Prediction')

plt.legend()

plt.show()

print('sumError=%f'%(sumError))

print('avgError=%f'%(sumError/count))
