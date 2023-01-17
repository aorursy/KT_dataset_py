from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import random
class Perceptron(object):

    

    def __init__(self, l_rate=0.1, n_iter=100):

        self.l_rate = l_rate

        self.n_iter = n_iter

        

    def train(self, x, y):

        

        # First element of the weight array is threshold

        # Weights and threshold initialized with zeros

        

        self.weight = np.zeros(x.shape[1]+1)

        self.costs = []

        

        for i in range(self.n_iter):

            cost = 0

            for xi, di in zip(x, d):

                y = self.predict(xi)

                error = di - y

                self.weight[1:] += self.l_rate * error * xi

                cost += (error**2) / 2.0

                

            self.costs.append(cost)

        

        return self

    

    # Activation function, returns 1 or -1

    def predict(self, x):

        return np.where(np.dot(self.weight[1:], x) + self.weight[0] >= 0.0, 1, -1)
# Creating data sets



def createSet(classSize):

    classA = np.random.choice(np.arange(0, maxX, factor), size=(classSize, 3))

    classB = np.random.choice(np.arange(minX, 0, factor), size=(classSize, 3))

    

    return classA, classB



# Variables to create data sets

classSize = 50

trainingSize = 40

minX = -1

maxX = 1

factor = 0.1



classA, classB = createSet(classSize)



# Training set including first 40 element of each class

trainingA = classA[:trainingSize]

trainingB = classB[:trainingSize]



# All inputs to train network 

x = np.concatenate((trainingA, trainingB), axis=0)



# d is the desired outputs for training sets

# It includes as; Class A = 1, Class B = -1

d = np.ones(trainingSize*2)

d[trainingSize:] = -1



# Testing inputs including last 10 element of each class

testA = classA[trainingSize:]

testB = classB[trainingSize:]



# Plotting part of training data

fig = plt.figure()

ax = Axes3D(fig)



ax.scatter(trainingA[:, 0], trainingA[:, 1], trainingA[:, 2], c='r', marker='o')

ax.scatter(trainingB[:, 0], trainingB[:, 1], trainingB[:, 2], c='b', marker='o')



ax.set_xlabel('X1')

ax.set_ylabel('X2')

ax.set_zlabel('X3')



plt.show()
l_rate = 0.01

n_iter = 100



# Initializing Percepton class object

model = Perceptron(l_rate, n_iter)

model.train(x, d)

weights = model.weight



print("Threshold: ", weights[0])

print("Weights: ", weights[1:])



plt.plot(range(1, len(model.costs) + 1), model.costs, marker='x')

plt.xlabel('Iteration')

plt.ylabel('Cost Function')

plt.show()
fig = plt.figure()

ax = Axes3D(fig)



ax.scatter(trainingA[:, 0], trainingA[:, 1], trainingA[:, 2], c='r', marker='o')

ax.scatter(trainingB[:, 0], trainingB[:, 1], trainingB[:, 2], c='b', marker='o')



[x1, x2] = np.meshgrid(np.arange(minX, maxX, factor), np.arange(minX, maxX, factor))

x3 = (weights[0] - (weights[1]*x1) - (weights[2]*x2)) / weights[3]

ax.plot_surface(x1, x2, x3, color='c', alpha=0.3)
testResultA = [model.predict(testA[i]).tolist() for i in range(len(testA))]

testResultB = [model.predict(testB[i]).tolist() for i in range(len(testB))]



print("Test Reults for 10 Class A Samples:\n", testResultA)



print("\n\nTest Reults for 10 Class B Samples:\n", testResultB)
fig = plt.figure()

ax = Axes3D(fig)



ax.scatter(testA[:, 0], testA[:, 1], testA[:, 2], c='r', marker='o')

ax.scatter(testB[:, 0], testB[:, 1], testB[:, 2], c='b', marker='o')



weights = model.weight



[x1, x2] = np.meshgrid(np.arange(minX, maxX, factor), np.arange(minX, maxX, factor))

x3 = (weights[0] - (weights[1]*x1) - (weights[2]*x2)) / weights[3]

ax.plot_surface(x1, x2, x3, color='c', alpha=0.3)