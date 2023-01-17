import pandas as pd

import numpy as np

import sklearn 

import statistics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
import os

os.listdir("../input/proxymeanstest-cr")
crtrain = pd.read_csv("../input/proxymeanstest-cr/train.csv",

          na_values = 'Na')
crtest = pd.read_csv("../input/proxymeanstest-cr/test.csv",

          na_values = 'Na')
crtrain.shape
crtrain.shape
for i in range(len(crtrain["Id"])):

    if crtrain.parentesco1[i] != 1:

        newtrain = newtrain.drop(i)
for i in range(len(crtest["Id"])):

    if crtest.parentesco1[i] != 1:

        newtest = newtest.drop(i)
newtrain.shape
newtest.shape

newtrain.v2a1 = newtrain.v2a1.fillna(value = 0, downcast=None)

newtest.v2a1 = newtest.v2a1.fillna(value = 0, downcast=None)

newtrain.v18q1 = newtrain.v18q1.fillna(value = 0, downcast=None)

newtest.v18q1 = newtest.v18q1.fillna(value = 0, downcast=None)

newtrain.qmobilephone = newtrain.qmobilephone.fillna(value = 0, downcast=None)

newtest.qmobilephone = newtest.qmobilephone.fillna(value = 0, downcast=None)
newtrain["epared"] = 2*newtrain["epared3"] + newtrain["epared2"]

newtrain["etecho"] = 2*newtrain["etecho3"] + newtrain["etecho2"]

newtrain["eviv"] = 2*newtrain["eviv3"] + newtrain["eviv2"]

newtest["epared"] = 2*newtest["epared3"] + newtest["epared2"]

newtest["etecho"] = 2*newtest["etecho3"] + newtest["etecho2"]

newtest["eviv"] = 2*newtest["eviv3"] + newtest["eviv2"]

Xtrain = newtrain[["v2a1","v18q1","hhsize","noelec","sanitario1","energcocinar1","elimbasu1","epared","etecho","eviv","area1","qmobilephone"]]

Ytrain = newtrain.Target
means =[]

for num in range(1,31):

    knn = KNeighborsClassifier(n_neighbors = num)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

    mean = statistics.mean(scores)

    means.append(mean) 

bestn = means.index(max(means))+1

bestn
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

scores
def percent(colum):

    return colum*100//float(sum(colum))

import matplotlib.pyplot as plt
newtrain["hhsize"].value_counts().plot(kind='bar')
pd.crosstab(newtrain["hhsize"],newtrain["Target"]).plot()