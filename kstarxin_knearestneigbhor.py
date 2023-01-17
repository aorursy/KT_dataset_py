# Run some setup code for this notebook.

from __future__ import print_function

import random

import numpy as np

import matplotlib.pyplot as plt





# This is a bit of magic to make matplotlib figures appear inline in the notebook

# rather than in a new window.

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



# Some more magic so that the notebook will reload external python modules;

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2
import pickle

import numpy as np

import random

import matplotlib.pyplot as plt

with open("../input/randomdata/random_data.pkl", "rb") as f:

    X_data, y_data = pickle.load(f)

X_data = np.array(X_data)

y_data = np.array(y_data)



X_train = X_data[:250]

y_train = y_data[:250]



X_test = X_data[250:]

y_test = y_data[250:]



plt.plot(X_train[:,0], X_train[:, 1], "o")

train_classified = []

for i in range(5): train_classified.append([])

for i, p in enumerate(X_train):

    train_classified[y_train[i]].append(p)



train_classified = [np.array(x) for x in train_classified]

train_classified = np.array(train_classified) 

plt.plot(train_classified[0][:,0], train_classified[0][:,1], 'o', color='red')

plt.plot(train_classified[1][:,0], train_classified[1][:,1], 'o', color='blue')

plt.plot(train_classified[2][:,0], train_classified[2][:,1], 'o', color='green')

plt.plot(train_classified[3][:,0], train_classified[3][:,1], 'o', color='orange')

plt.plot(train_classified[4][:,0], train_classified[4][:,1], 'o', color='yellow')

plt.plot(X_test[:,0], X_test[:, 1], 'o', color='black')

plt.show()

np.random.seed(2)



class KNearestNeighbor(object):

    def __init__(self):

        pass

    

    def train(self, X_train, y_train):

        self.X_train = X_train

        self.y_train = y_train

    

    def compute_distances_naive(self, X):

        num_test = X.shape[0]

        num_train = self.X_train.shape[0]

        dists = np.zeros((num_test, num_train))

        for i in range(num_test):

            for j in range(num_train):

                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

        return dists

    

    def compute_distances(self, X):

        num_test = X.shape[0]

        num_train = self.X_train.shape[0]

        Xtrain = np.expand_dims(np.sum(self.X_train ** 2, axis=1), axis=0)

        Xtest = np.expand_dims(np.sum(X ** 2, axis = 1), axis=1)

        dists = np.sqrt(Xtrain + Xtest - 2 * np.dot(X, self.X_train.T))

        return dists

    

    def predict_labels(self, dists, category = 10, k = 1):

        num_test, num_train = dists.shape

        y_pred = np.zeros(num_test)

        for i in range(num_test):

            closest_y = self.y_train[dists[i].argsort()[:k]]

            bucket = np.zeros(category)

            for j in closest_y:

                bucket[j] += 1

                y_pred[i] = np.argmax(bucket)

        return y_pred

    

    def predict(self, X, category=10, k=1):

        dists = self.compute_distances(X)

        return self.predict_labels(dists, category, k=k)

        
classifier = KNearestNeighbor()

classifier.train(X_train, y_train)

y_pred = classifier.predict(X_test, 5)



plt.plot(train_classified[0][:,0], train_classified[0][:,1], 'o', color='red')

plt.plot(train_classified[1][:,0], train_classified[1][:,1], 'o', color='blue')

plt.plot(train_classified[2][:,0], train_classified[2][:,1], 'o', color='green')

plt.plot(train_classified[3][:,0], train_classified[3][:,1], 'o', color='orange')

plt.plot(train_classified[4][:,0], train_classified[4][:,1], 'o', color='yellow')



test_classified = [[] for i in range(5)]

for i, p in enumerate(X_test):

    test_classified[int(y_pred[i])].append(p)



test_classified = [np.array(x) for x in test_classified]

tesst_classified = np.array(test_classified) 





plt.plot(test_classified[0][:,0], test_classified[0][:,1], 'o', color='red')

plt.plot(test_classified[1][:,0], test_classified[1][:,1], 'o', color='blue')

plt.plot(test_classified[2][:,0], test_classified[2][:,1], 'o', color='green')

plt.plot(test_classified[3][:,0], test_classified[3][:,1], 'o', color='orange')

plt.plot(test_classified[4][:,0], test_classified[4][:,1], 'o', color='yellow')



plt.show()
import os

import pandas

with open("../input/fashionmnist/fashion-mnist_test.csv", "r") as f:

    test_data = pandas.read_csv(f).values

with open("../input/fashionmnist/fashion-mnist_train.csv", "r") as f:

    train_data = pandas.read_csv(f).values



X_train = train_data[:, 1:].astype(np.float32)

y_train = train_data[:, 0]



X_test = test_data[:, 1:].astype(np.float32)

y_test = test_data[:, 0]

    

#As a sanity check, we print out the size of the training and test data.

print('Training data shape: ', X_train.shape)

print('Training labels shape: ', y_train.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)
# Visualize some examples from the dataset.

# We show a few examples of training images from each class.

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

num_classes = len(classes)

samples_per_class = 7

for y, cls in enumerate(classes):

    idxs = np.flatnonzero(y_train == y)

    idxs = np.random.choice(idxs, samples_per_class, replace=False)

    for i, idx in enumerate(idxs):

        plt_idx = i * num_classes + y + 1

        plt.subplot(samples_per_class, num_classes, plt_idx)

        plt.imshow(X_train[idx].reshape((28,28)),cmap=plt.cm.gray)

        plt.axis('off')

        if i == 0:

            plt.title(cls)

plt.show()
# Subsample the data for more efficient code execution in this exercise

num_training = 5000

mask = list(range(num_training))

X_train = X_train[mask]

y_train = y_train[mask]



num_test = 500

mask = list(range(num_test))

X_test = X_test[mask]

y_test = y_test[mask]
# Reshape the image data into rows

X_train = np.reshape(X_train / 255.0, (X_train.shape[0], -1))

X_test = np.reshape(X_test / 255.0, (X_test.shape[0], -1))

print(X_train.shape, X_test.shape)
# Create a kNN classifier instance. 

# Remember that training a kNN classifier is a noop: 

# the Classifier simply remembers the data and does no further processing 

classifier = KNearestNeighbor()

classifier.train(X_train, y_train)

# While encounter 'No module named 'past'' : run pip install future
# Open MS325/classifiers/k_nearest_neighbor.py and implement

# compute_distances_two_loops.



# Test your implementation:

dists = classifier.compute_distances(X_test)

print(dists.shape)
# Now implement the function predict_labels and run the code below:

# We use k = 1 (which is Nearest Neighbor).

y_test_pred = classifier.predict_labels(dists, k=1)



# Compute and print the fraction of correctly predicted examples

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
y_test_pred = classifier.predict_labels(dists, k=5)

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
# Let's compare how fast the implementations are

def time_function(f, *args):

    """

    Call a function f with args and return the time (in seconds) that it took to execute.

    """

    import time

    tic = time.time()

    f(*args)

    toc = time.time()

    return toc - tic



two_loop_time = time_function(classifier.compute_distances_naive, X_test)

print('Naive version took %f seconds' % two_loop_time)



no_loop_time = time_function(classifier.compute_distances, X_test)

print('Good version took %f seconds' % no_loop_time)



# you should see significantly faster performance with the fully vectorized implementation
import numpy as np

num_folds = 5

k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]



X_train_folds = []

y_train_folds = []

################################################################################

# TODO:                                                                        #

# Split up the training data into folds. After splitting, X_train_folds and    #

# y_train_folds should each be lists of length num_folds, where                #

# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #

# Hint: Look up the numpy array_split function.                                #

################################################################################

X_train_folds = np.array_split(np.array(X_train), num_folds)

y_train_folds = np.array_split(np.array(y_train), num_folds)



################################################################################

#                                 END OF YOUR CODE                             #

################################################################################



# A dictionary holding the accuracies for different values of k that we find

# when running cross-validation. After running cross-validation,

# k_to_accuracies[k] should be a list of length num_folds giving the different

# accuracy values that we found when using that value of k.

k_to_accuracies = {}





################################################################################

# TODO:                                                                        #

# Perform k-fold cross validation to find the best value of k. For each        #

# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #

# where in each case you use all but one of the folds as training data and the #

# last fold as a validation set. Store the accuracies for all fold and all     #

# values of k in the k_to_accuracies dictionary.                               #

################################################################################

for k in k_choices:

    accl = []

    for i in range(num_folds):

        Xctest = np.array(X_train_folds[i])

        yctest = np.array(y_train_folds[i])

        Xctrain = []

        yctrain = []

        for j in range(num_folds):

            if j != i:

                Xctrain.append(X_train_folds[j])

                yctrain.append(y_train_folds[j])

        Xctrain = np.concatenate(Xctrain)

        yctrain = np.concatenate(yctrain)

        classifier.train(Xctrain, yctrain)

        dists = classifier.compute_distances(Xctest)

        ycpred = classifier.predict_labels(dists, 10, k)

        acc = np.sum(ycpred == yctest) / yctest.shape[0]

        accl.append(acc)

    k_to_accuracies[k] = np.array(accl)

################################################################################

#                                 END OF YOUR CODE                             #

################################################################################



# Print out the computed accuracies

for k in sorted(k_to_accuracies):

    for accuracy in k_to_accuracies[k]:

        print('k = %d, accuracy = %f' % (k, accuracy))
# plot the raw observations

for k in k_choices:

    accuracies = k_to_accuracies[k]

    plt.scatter([k] * len(accuracies), accuracies)



# plot the trend line with error bars that correspond to standard deviation

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])

accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])

plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)

plt.title('Cross-validation on k')

plt.xlabel('k')

plt.ylabel('Cross-validation accuracy')

plt.show()
# Based on the cross-validation results above, choose the best value for k,   

# retrain the classifier using all the training data, and test it on the test

# data. You should be able to get above 75% accuracy on the test data.

best_k = 3



classifier = KNearestNeighbor()

classifier.train(X_train, y_train)

y_test_pred = classifier.predict(X_test, k=best_k)



# Compute and display the accuracy

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))