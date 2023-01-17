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

# For E epochs, 
E = 15
errors = []
for epoch in range(E):
    err = 0 #reset the error counter
    # For each handwritten digit in training set,
    for i, x in enumerate(traindata):
        dp=[] #create a new container for the calculated dot products
        # For each digit class (0-9)
        for w in weights:
            dotproduct = np.dot(x,w) #take the dot product of the weight and the data
            dp.append(dotproduct) #add the dot product to the list of dot products
        guess = np.argmax(dp) #take the largest dot product and make that the guessed digit class
        actual = trainlabels[i]
        # If the guess was wrong, update the weight vectors
        if guess != actual:
            weights[guess] = weights[guess] - x #update the incorrect (guessed) weight vector
            weights[actual] = weights[actual] + x #update the correct weight vector
            err += 1
    errors.append(err/42000) #track the error after each pass through the training set
plt.plot(list(range(0,E)),errors) #plot the error after all training epochs


testdata = np.asmatrix(test)
guesses = []
for i, z in enumerate(testdata):
    dp=[] #create a new container for the calculated dot products
    # For each digit class (0-9)
    for w in weights:
        dotproduct = np.dot(z,w) #take the dot product of the weight and the data
        dp.append(dotproduct) #add the dot product to the list of dot products
    guess = np.argmax(dp) #take the largest dot product and make that the guessed digit class
    guesses.append(guess)
f = open('percep1.csv', 'w')
i = 1
f.write('ImageId,Label\n')
for g in guesses:
    foo = str(i) + ',' + str(g) + '\n'
    f.write(foo)
    i += 1
f.close()
print(i)



