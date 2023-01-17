# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/Iris.csv")

print (df.head())

clas = []

for i in range(len(df["Species"])):

    if df["Species"][i] == "Iris-setosa":

        clas.append(0)

    elif df["Species"][i] == "Iris-versicolor":

        clas.append(1)

    else: clas.append(2)

df["category"]=clas

print (df.describe())

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

grr = pd.scatter_matrix(df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "category"]], 

                        c=df["category"], figsize=(15, 15), marker='o',

                            hist_kwds={'bins': 20}, s=60, alpha=.8)
#trying a simple logistic regression approach

predictors = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

from sklearn.linear_model import LogisticRegression

alg = LogisticRegression(random_state=1)

alg.fit(df[predictors], df["category"])

predictions =  alg.predict(df[predictors])

tot = 0

print ("Mismatched cases: ")

print ("Predicted Value vs Actual Value")

for i in range(len(predictions)):

    if predictions[i]==df["category"][i]: tot+=1

    else:

        print (predictions[i], df["category"][i])

tot = tot*1.0/len(predictions)

print ("Accuracy of Logistic Regression:")

print (tot)

#trying with adding a new feature

df["petalArea"] = df["PetalLengthCm"]*df["PetalWidthCm"]

predictors.append("petalArea")

alg.fit(df[predictors], df["category"])

predictions =  alg.predict(df[predictors])

tot = 0

print ("Mismatched cases: ")

print ("Predicted Value vs Actual Value")

for i in range(len(predictions)):

    if predictions[i]==df["category"][i]: tot+=1

    else:

        print (predictions[i], df["category"][i])

tot = tot*1.0/len(predictions)

print ("Accuracy of Logistic Regression:")

print (tot)

# I guess logistic regression won't be going anywhere more than 96% accuracy
# trying out the k means clustering algorithm

#trying out k means algorithm using k=3

from random import randint

from sklearn import cluster, datasets

k_means = cluster.KMeans(n_clusters=3)

k_means.fit(df[predictors])

tot=0

temp = k_means.labels_

t2 = []

for i in temp:

    if i==temp[0]:

        t2.append(0)

    elif i==temp[55]:

        t2.append(1)

    else: t2.append(2)

tot=0

# print df["category"]

for i in range(len(temp)):

    if t2[i]==df["category"][i]: tot+=1

    else:

        print (t2[i], df["category"][i])

tot = tot*1.0/len(predictions)

print ("Accuracy of K means:")

print (tot)
#we see that logistic regression is working better than K means!

# my next approach will be using random forests.

#using random forests

from sklearn import cross_validation

from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=2, min_samples_leaf=2)

scores = cross_validation.cross_val_score(alg,df[predictors], df["category"], cv=3)

print (scores.mean())

print ("Random forests works better than k means or logistic regression")
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(df[predictors], df[category])