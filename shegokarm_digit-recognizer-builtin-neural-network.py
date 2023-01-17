# Importing libraries
import numpy as np
import pandas as pd
# Importing data 
traindata = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")
# Looking at top 5 lines from the train data
traindata.head()
# Looking at top 5 lines from the test data
testdata.head()
# Dimensions of the train and test data
print("Train data: ", traindata.shape)
print("\nTest data: ", testdata.shape)
# Importing libraries for plotting images
import matplotlib.pyplot as plt
import math

# Plotting images
fig = plt.figure()

for i in range(1,26):
    ax = fig.add_subplot(5,5,i)
    data = traindata.iloc[i, 1:len(traindata)].values
    grid = data.reshape(int(math.sqrt(len(traindata.iloc[1, 1:len(traindata)]))),int(math.sqrt(len(traindata.iloc[1, 1:len(traindata)]))))
    ax.imshow(grid)
    
plt.savefig("Digits.png")    
# Importing neural network library
from sklearn.neural_network import MLPClassifier
# Separating label from the original data to train neural network
trainy = traindata.iloc[:, 0]
trainx = traindata.iloc[:, 1:]
# Training model
clf = MLPClassifier()
clf.fit(trainx, trainy)
# Importing libraries
from sklearn import metrics

predicted = clf.predict(trainx)
expected = trainy

# Measuring accuracy of training set
print("Training model accuracy: ", metrics.accuracy_score(expected, predicted))
# Predicting test data using train model
result = pd.DataFrame(clf.predict(testdata), columns = ["Label"])

# Resetting index
result.reset_index(inplace = True)

# Renaming column names
result.rename(columns = {'index': 'ImageId'}, inplace = True)
result['ImageId'] = result['ImageId'] + 1

# output
result.to_csv('result.csv', index = False)