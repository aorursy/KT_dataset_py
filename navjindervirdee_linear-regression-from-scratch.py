# pandas for data processing

import pandas as pd



# numpy for mathematical calculations

import numpy as np



# matplotlib for visualization

import matplotlib.pyplot as plt
# path of the dataset

path = r'../input/train.csv'



# read the dataset

data = pd.read_csv(path)

print("Shape of the dataset = {}".format(data.shape))



# lets see first 5 rows of the dataset

data.head(5)
# drop any rows with 'nan' values

data = data.dropna()

print("Shape of the dataset = {}".format(data.shape))
# let's visualize the dataset

plt.plot(data.x[0:500], data.y[0:500], 'x')

plt.title("X vs Y")

plt.xlabel("x")

plt.ylabel("y")
# training dataset and labels

train_dataset = np.array(data.x[0:500]).reshape(500,1)

train_labels  = np.array(data.y[0:500]).reshape(500,1)



# valid dataset and labels

valid_dataset = np.array(data.x[500:700]).reshape(199,1)

valid_labels  = np.array(data.y[500:700]).reshape(199,1)



# print the shapes

print("Train Dataset Shape = {}".format(train_dataset.shape))

print("Train Labels  Shape = {}".format(train_labels.shape))

print("Valid Dataset Shape = {}".format(valid_dataset.shape))

print("Valid Labels  Shape = {}".format(valid_labels.shape))
def forward_propagation(train_dataset, parameters):

    w = parameters['w']

    b = parameters['b']

    predictions = np.multiply(w, train_dataset) + b

    return predictions
def cost_function(predictions, train_labels):

    cost = np.mean((train_labels - predictions) ** 2) * 0.5

    return cost
def backward_propagation(train_dataset, train_labels, predictions):

    derivatives = dict()

    df = (train_labels - predictions) * -1

    dw = np.mean(np.multiply(train_dataset, df))

    db = np.mean(df)

    derivatives['dw'] = dw

    derivatives['db'] = db

    return derivatives
def update_parameters(parameters, derivatives, learning_rate):

    parameters['w'] = parameters['w'] - learning_rate * derivatives['dw']

    parameters['b'] = parameters['b'] - learning_rate * derivatives['db']

    return parameters
def train(train_dataset, train_labels, learning_rate, iters = 10):

    #random parameters

    parameters = dict()

    parameters["w"] = np.random.uniform(0,1) * -1

    parameters["b"] = np.random.uniform(0,1) * -1

    

    plt.figure()

    

    #loss

    loss = list()

    

    #iterate

    for i in range(iters):

        

        #forward propagation

        predictions = forward_propagation(train_dataset, parameters)

        

        #cost function

        cost = cost_function(predictions, train_labels)

        

        #append loss and print

        loss.append(cost)

        print("Iteration = {}, Loss = {}".format(i+1, cost))

        

        #plot function

        plt.plot(train_dataset, train_labels, 'x')

        plt.plot(train_dataset, predictions, 'o')

        plt.show()

        

        #back propagation

        derivatives = backward_propagation(train_dataset, train_labels, predictions)

        

        #update parameters

        parameters = update_parameters(parameters, derivatives, learning_rate)

        

    return parameters, loss
parameters,loss = train(train_dataset, train_labels, 0.0001, 20)
valid_predictions = valid_dataset * parameters["w"] + parameters["b"]

plt.figure()

plt.plot(valid_dataset, valid_labels, 'x')

plt.plot(valid_dataset, valid_predictions, 'o')

plt.show()
#cost for valid dataset

cost_function(valid_predictions, valid_labels)