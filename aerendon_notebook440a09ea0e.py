import pandas as pd

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import numpy as np

import time

from sklearn import tree
dataset = pd.read_csv("../input/Iris.csv", index_col=0)



train = dataset._slice(slice(0, 104))

test = dataset._slice(slice(105, 150))



trainClass = train['Species']

trainFeatures = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]



testClass = test['Species']

testFeatures = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
plt.rcParams["figure.figsize"] = [16, 9]

train.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.show()
train.hist()

plt.show()
scatter_matrix(train)

plt.show()
start_time = time.clock()

clf = tree.DecisionTreeClassifier()

clf = clf.fit(trainFeatures, trainClass)

print(time.clock() - start_time, "seconds")
accuracy = clf.score(testFeatures, testClass)

print("Accuracy is", accuracy)