%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
# Store the labels in a vector
trainlabels = train.label

# Store the data in a matrix (without labels)
traindata = np.asmatrix(train.ix[:,'pixel0':])
traindata.shape

# Visualize a single digit (one row from the training data matrix)
samplerow = traindata[3:4]#get one row from training data
samplerow = np.reshape(samplerow,(28,28))#reshape it to a 28*28 grid
print("A sample digit from the dataset:")
plt.imshow(samplerow, cmap="hot")

# Initialize the weight matrix (one row per class)
weights = np.zeros((10,784))
print(weights.shape)
#Calculate the Priors
priors = np.zeros(10)
for y in trainlabels: #for each digit, update that class count
    priors[y] += 1
for i in range(len(priors)):
    priors[i] = priors[i]/len(trainlabels)
