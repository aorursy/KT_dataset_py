# 做一些准备性的工作，print_function目的是为了要兼容py2，py3的输出。

from __future__ import print_function

import sys

sys.path.append('/kaggle/input/ms326-knn')

import random

import numpy as np

import matplotlib.pyplot as plt



# 让图片生成在notebook中而不是生成新的窗口

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'





#作用：在调试的过程中，如果代码发生更新，实现ipython中引用的模块也能自动更新。

# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

%load_ext autoreload

%autoreload 2
# Load the raw Fashion_mnist data.

import os

import pandas

with open("../input/fashionmnist/fashion-mnist_test.csv", "r") as f:

    test_data = pandas.read_csv(f).values

with open("../input/fashionmnist/fashion-mnist_train.csv", "r") as f:

    train_data = pandas.read_csv(f).values

    

X_train = train_data[:, 1:].astype(np.float32)

y_train = train_data[:, 0]



X_test = train_data[:, 1:].astype(np.float32)

y_test = train_data[:, 0]



# As a sanity check, we print out the size of the training and test data.

print('Training data shape: ', X_train.shape)

print('Training labels shape: ', y_train.shape)

print('Test data shape: ', X_test.shape)

print('Test labels shape: ', y_test.shape)
# np.flatnonzero 矩阵扁平化后返回非零元素的位置

# numpy.random.choice(a, size=None, replace=True, p=None) 随机选取a中的值

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

num_classes = len(classes)

samples_per_class = 7

for y, cls in enumerate(classes):

    idxs = np.flatnonzero(y_train == y)

    idxs = np.random.choice(idxs, samples_per_class, replace=False)

    for i, idx in enumerate(idxs):

        plt_idx = i * num_classes + y + 1

        plt.subplot(samples_per_class, num_classes, plt_idx)

        plt.subplots_adjust(left=0, bottom=None, right=None, top=None,

                wspace=0.1, hspace=0.1)

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

X_train = np.reshape(X_train, (X_train.shape[0], -1))

X_train = np.array(X_train, dtype = np.int32)

X_test = np.reshape(X_test, (X_test.shape[0], -1))

X_test = np.array(X_test, dtype = np.int32)

print(X_train.shape, X_test.shape)
from k_nearest_neighbor import KNearestNeighbor



# Create a kNN classifier instance. 

# Remember that training a kNN classifier is a noop: 

# the Classifier simply remembers the data and does no further processing 

classifier = KNearestNeighbor()

classifier.train(X_train, y_train)

# While encounter 'No module named 'past'' : run pip install future
# Open MS326/classifiers/k_nearest_neighbor.py and implement

# compute_distances_two_loops.

def compute_distances_two_loops(self, X):

    num_test = X.shape[0]

    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train))

    for i in xrange(num_test):

        for j in xrange(num_train):

            dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

    return dists





# Test your implementation:

dists = classifier.compute_distances_two_loops(X_test)

print(dists.shape)
# We can visualize the distance matrix: each row is a single test example and

# its distances to training examples

plt.imshow(dists, interpolation='none')

plt.show()
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
# Now lets speed up distance matrix computation by using partial vectorization

# with one loop. Implement the function compute_distances_one_loop and run the

# code below:



# python Broadcasting机制

def compute_distances_one_loop(self, X):

    num_test = X.shape[0]

    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train))

    for i in xrange(num_test):

        dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis = 1))

    return dists



dists_one = classifier.compute_distances_one_loop(X_test)



# 我们可以用 Frobenius 范数去判断我们是否求解正确了. 

# In case you haven't seen it before, the Frobenius norm of two matrices is the square

# root of the squared sum of differences of all elements; in other words, reshape

# the matrices into vectors and compute the Euclidean distance between them.

difference = np.linalg.norm(dists - dists_one, ord='fro')

print('Difference was: %f' % (difference, ))

if difference < 0.001:

    print('Good! The distance matrices are the same')

else:

    print('Uh-oh! The distance matrices are different')
# Now implement the fully vectorized version inside compute_distances_no_loops

# and run the code

def compute_distances_no_loops(self, X):

    num_test = X.shape[0]

    num_train = self.X_train.shape[0]

    dists = np.zeros((num_test, num_train)) 

    #########################################################################

    # HINT: Try to formulate the l2 distance using matrix multiplication    #

    #       and two broadcast sums                                          #

    # Attention: square(x1-x2) = square(x1) + square(x2) - 2 * x1 * x2      #

    #########################################################################

    x1_x2 = np.dot(X, self.X_train.T)  # num_test * num_train

    test_square = np.sum(np.square(X), axis=1).reshape(-1,1) # num_test * 1

    train_square = np.sum(np.square(self.X_train.T), axis=0).reshape(1,-1) # 1 * num_trian

    dists = np.sqrt(-2 * x1_x2 + test_square + train_square) #broadcast

    return dists





dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:

difference = np.linalg.norm(dists - dists_two, ord='fro')

print('Difference was: %f' % (difference, ))

if difference < 0.001:

    print('Good! The distance matrices are the same')

else:

    print('Uh-oh! The distance matrices are different')
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



two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)

print('Two loop version took %f seconds' % two_loop_time)



one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)

print('One loop version took %f seconds' % one_loop_time)



no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)

print('No loop version took %f seconds' % no_loop_time)



# you should see significantly faster performance with the fully vectorized implementation
num_folds = 5

k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]



X_train_folds = []

y_train_folds = []

################################################################################                                                                       #

# Split up the training data into folds. After splitting, X_train_folds and    #

# y_train_folds should each be lists of length num_folds, where                #

# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #

# Hint: Look up the numpy array_split function.                                #

################################################################################

y_train_ = y_train.reshape(-1, 1)

X_train_folds , y_train_folds = np.array_split(X_train, num_folds), np.array_split(y_train_, num_folds)



# A dictionary holding the accuracies for different values of k that we find

# when running cross-validation. After running cross-validation,

# k_to_accuracies[k] should be a list of length num_folds giving the different

# accuracy values that we found when using that value of k.

k_to_accuracies = {}



################################################################################

# Perform k-fold cross validation to find the best value of k. For each        #

# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #

# where in each case you use all but one of the folds as training data and the #

# last fold as a validation set. Store the accuracies for all fold and all     #

# values of k in the k_to_accuracies dictionary.                               #

################################################################################

for k in k_choices:

    k_to_accuracies.setdefault(k, [])



for i in range(num_folds):

    classifier = KNearestNeighbor()

    X_val_train = np.vstack(X_train_folds[0:i] + X_train_folds[i+1:])

    y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i+1:])[:,0]

    classifier.train(X_val_train, y_val_train)

    for k in k_choices:

        y_val_pred = classifier.predict(X_train_folds[i], k=k)

        num_correct = np.sum(y_val_pred == y_train_folds[i][:,0])

        accuracy = float(num_correct) / len(y_val_pred)

        k_to_accuracies[k] = k_to_accuracies[k] + [accuracy]

    

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

# data. You should be able to get above 83% accuracy on the test data.

best_k = 15



classifier = KNearestNeighbor()

classifier.train(X_train, y_train)

y_test_pred = classifier.predict(X_test, k=best_k)



# Compute and display the accuracy

num_correct = np.sum(y_test_pred == y_test)

accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))