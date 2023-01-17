# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Load the training data
df1=pd.read_csv('../input/train.csv', sep=',',header=None)
df2=pd.read_csv('../input/test.csv', sep=',',header=None)
train_data = df1.values
train_data = np.delete(train_data, 0, axis=0) # deleting column names
train_labels = train_data[:,0]
train_data = np.delete(train_data, 0, axis=1) # deleting the label column
test_data = df2.values
test_data = np.delete(test_data, 0, axis=0) # deleting column names

#Printing out the dimensions
print("Training dataset dimensions: ", np.shape(train_data))
print("Number of training labels: ", len(train_labels))
print("Testing dataset dimensions: ", np.shape(test_data))
train_labels = list(map(int, train_labels))
np.shape(test_data)

## Compute the number of examples of each digit
train_digits, train_counts = np.unique(train_labels[1:], return_counts = True)
print("Training set distribution:")
print(dict(zip(train_digits, train_counts)))
## Define a function that displays a digit given its vector representation
def show_digit(x):
    plt.axis("off")
    x = np.asarray(list(map(int, x)))
    plt.imshow(x.reshape((28,28)), cmap = plt.cm.gray)
    plt.show()
    return

## Define a function that takes an index into a particular data set ("train" or "test") and displays that image.
def vis_image(index, dataset="train"):
    if (dataset=='train'):
        show_digit(train_data[index,])
        label = train_labels[index]
        print("Label "+ str(label))
    else:
        show_digit(test_data[index,])
        #label = test_labels[index]
    #print("Label "+ str(label))
    return

## View the first data point in the training set
vis_image(56, "train")
##computes squared euclidean distance between two vectors
def squared_dist(x, y):
    x = np.asarray(list(map(float, x)))
    y = np.asarray(list(map(float, y)))
    return np.sum(np.square(x-y))

## Compute distance between a six and a two in our training set
print ("Distance from 6 to 2: ", squared_dist(train_data[266,],train_data[22,]))
## Takes a vector x and returns the index of its nearest neighbour in train_data
def find_NN(x):
    #compute distances from x to every row in train_data
    distances = [squared_dist(x, train_data[i,]) for i in range(len(train_labels))]
    #Get the index of the smallest distance
    return np.argmin(distances)

## Takes a vector x and returns the class of its nearest neighbour in train_data
def NN_classifier(x):
    #Get the index of the nearest neighbour
    index = find_NN(x)
    # Return its class
    return train_labels[index]

## A sample case
print ("A sample case:")
print ("NN classification: ", NN_classifier(test_data[39,]))
print ("The test image: ")
vis_image(39, "test")
print ("The corresponding nearest neighbour image: ")
vis_image(find_NN(test_data[39,]), "train")

## Predict on each test data point (and time it!)
t_before = time.time()
test_predictions = [NN_classifier(test_data[i,]) for i in range(10)]
t_after = time.time()

## Compute the error
#err_positions = np.not_equal(test_predictions, test_labels)
#error = float(np.sum(err_positions))/len(test_labels)

#print("Error of nearest neighbor classifier: ", error)
print("Classification time (seconds): ", t_after - t_before)
from sklearn.neighbors import BallTree

# Build nearest neighbour structure on training data
t_before = time.time()
ball_tree = BallTree(train_data)
t_after = time.time()

## Get nearest neighbor predictions on testing data
t_before = time.time()
test_neighbours = np.squeeze(ball_tree.query(test_data, k=1, return_distance=False))
#ball_tree_predictions = train_labels[test_neighbours]
t_after = time.time()

## Compute testing time
t_testing = t_after - t_before
print("Time to classify test set (seconds): ", t_testing)

ball_tree_predictions = []
for i in range(28000):
    ball_tree_predictions.append(train_labels[test_neighbours[i]])
ball_tree_predictions = []
for i in range(28000):
    ball_tree_predictions.append(train_labels[test_neighbours[i]])

imid = [i for i in range(1,28001)]
name = [('ImageId', 'Label')]
dat = list(zip(imid, ball_tree_predictions))
dat = name + dat
import csv

with open('sub.csv', "w") as f:
    writer = csv.writer(f)
    for row in dat:
        writer.writerow(row)
