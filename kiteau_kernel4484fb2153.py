import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import ast
# Training data

train = pd.read_csv("/path/train.csv")

# Test data

test = pd.read_csv("/path/test.csv")



train.head()
# Prepare training data dataframe.

# Use 7 features for simple predictions

trainX = train[['belongs_to_collection', 'budget', 'genres', 'original_language' ,'popularity', 'runtime', 'status']]

# Fix belongs_to_collection column

for i in range(len(trainX['belongs_to_collection'])):

    if trainX['belongs_to_collection'][i] != trainX['belongs_to_collection'][i]:

        trainX['belongs_to_collection'][i] = 0

    else: trainX['belongs_to_collection'][i] = 1

trainX['belongs_to_collection'] = pd.to_numeric(trainX['belongs_to_collection'])



# Fix genres column

for i in range(len(trainX['genres'])):

    if trainX['genres'][i] != trainX['genres'][i]:

        trainX['genres'][i] = 0

    else:

        curDict = ast.literal_eval(trainX['genres'][i])

        trainX['genres'][i] = len(curDict)



# For every unique entry in a feature that isn't numerical, create a column

trainX = pd.get_dummies(trainX)

# Replace every entry 'nan' with a zero

trainX = trainX.fillna(0)

trainX.head()
# Prepare test data dataframe.

testX = test[['belongs_to_collection', 'budget', 'genres', 'original_language' ,'popularity', 'runtime', 'status']]



# Fix belongs_to_collection column

for i in range(len(testX['belongs_to_collection'])):

    if testX['belongs_to_collection'][i] != testX['belongs_to_collection'][i]:

        testX['belongs_to_collection'][i] = 0

    else: testX['belongs_to_collection'][i] = 1

testX['belongs_to_collection'] = pd.to_numeric(testX['belongs_to_collection'])



# Fix genres column

for i in range(len(testX['genres'])):

    if testX['genres'][i] != testX['genres'][i]:

        testX['genres'][i] = 0

    else:

        curDict = ast.literal_eval(testX['genres'][i])

        testX['genres'][i] = len(curDict)



# For every unique entry in a feature that isn't numerical, create a column

# Every entry will have a column for every unique feature value, but only its own

# feature value will be 1, others will be 0

testX = pd.get_dummies(testX)

# Replace every entry 'nan' with a zero

testX = testX.fillna(0)

testX.head()
# Extract target variable from training set

trainY = train[['revenue']]

trainY
# Must create any column that is unique to train/test data in other data too.

# For example, there might be language entry 'GE' in test data, but not in train data,

# which may lead to bugs

trainX, testX = trainX.align(testX, join='outer', axis=1, fill_value=0)
# Scale features by normalization to avoid overflow and bias towards columns

def scaleFeature(dataFrame, column, trainX):

    stdVal = np.nanstd(trainX[column])

    meanVal = np.mean(trainX[column])

    dataFrame[column] = (dataFrame[column] - meanVal)/stdVal
# Scale test data values

scaleFeature(testX, 'budget', trainX)

scaleFeature(testX, 'popularity', trainX)

scaleFeature(testX, 'runtime', trainX)

testX.insert( 0, 'Ones', 1)

testX
# Scale train data features

scaleFeature(trainX, 'budget', trainX)

scaleFeature(trainX, 'popularity', trainX)

scaleFeature(trainX, 'runtime', trainX)

trainX.insert( 0, 'Ones', 1)

trainX
trainX = trainX.fillna(0)

testX = testX.fillna(0)
# Scale revenue from 0 to 1

trainY['revenue'] = np.log(trainY['revenue'])

rangeVal = max(trainY['revenue']) - min(trainY['revenue'])

minVal = min(trainY['revenue'])

trainY['revenue'] = (trainY['revenue'] - minVal)/rangeVal

trainY
# Get value matrixes from the dataframes

trainXVals = np.matrix(trainX.values)

trainYVals = np.matrix(trainY.values)

testXVals  = np.matrix(testX.values)

# Initialize theta with value amount equal to number of columns in trainX

theta = np.matrix(np.zeros(trainXVals.shape[1]))
# Calculate theta using normal equation

xTrans = np.transpose(trainXVals)

xTransDotX = xTrans.dot(trainXVals)

temp1 = np.linalg.pinv(xTransDotX)



temp2 = xTrans.dot(trainYVals)

g = temp1.dot(temp2)

g = g.transpose()
newRes = testX.dot(g.transpose())

newRes
newRes[0] = newRes[0].fillna(0)

newRes[0]
# Undo scaling

newRes = newRes * rangeVal

newRes = newRes + minVal

newRes
newRes = np.exp(newRes)

newRes
indArr = []

for i in range(3001, 3001 + 4398):

    indArr.append(i)

newRes.insert(0, 'id', indArr, 1)

newRes.columns = ['id', 'revenue']

newRes
newRes.to_csv('/path/solution.csv', index=False)