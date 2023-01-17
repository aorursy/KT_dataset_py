#imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



#helpers

sigLev = 3

sns.set_style("dark")
#load in dataset

trainFrame = pd.read_csv("../input/train.csv")

testFrame = pd.read_csv("../input/test.csv")
trainFrame.shape
trainFrame.dtypes
trainFrame.isnull().sum()
numMissing = trainFrame.isnull().sum()

numWithMissingObs = numMissing[numMissing > 0].shape[0]

print(numWithMissingObs)
colsWithMissingObs = numMissing[numMissing > 0].index

filteredTrainFrame = trainFrame.drop(colsWithMissingObs,axis = 1)
filteredTrainFrame.shape
sdVec = filteredTrainFrame.std()

sdVec = sdVec.sort_values()

sdVec
timeCountFrame = filteredTrainFrame.groupby("timestamp")["timestamp"].count()

#then plot

timeCountFrame.plot()

plt.xlabel("Time Stamp")

plt.ylabel("Count")

plt.title("Observations over Time For Training Data")
timeCountFrame = testFrame.groupby("timestamp")["timestamp"].count()

#then plot

timeCountFrame.plot()

plt.xlabel("Time Stamp")

plt.ylabel("Count")

plt.title("Observations over Time For Test Data")
filteredTrainFrame = filteredTrainFrame.drop("timestamp",axis = 1)
priceDocVec = filteredTrainFrame["price_doc"]

filteredTrainFrame = filteredTrainFrame.drop("price_doc",axis = 1)
from sklearn.decomposition import PCA

testPCA = PCA()

testPCA.fit(filteredTrainFrame)