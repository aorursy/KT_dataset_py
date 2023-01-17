# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")
iris.info()
iris.head()
iris.describe()
iris.corr()
iris.groupby('Species').mean()['SepalLengthCm'].plot(kind='bar',color='g',figsize=(10,5))
sns.boxplot(x="Species", y="SepalLengthCm", data=iris)
sns.stripplot(x="Species", y="SepalWidthCm", data=iris, jitter=True);
iris.groupby('Species').mean()['PetalLengthCm'].plot(kind='barh',color='g',figsize=(10,5))
sns.violinplot(x="Species", y="PetalLengthCm", data=iris)
sns.stripplot(x="Species", y="PetalWidthCm", data=iris);
#BASED ON SEPAL

sns.swarmplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=iris);
#BASED ON PETAL

sns.swarmplot(x="PetalLengthCm", y="PetalWidthCm", hue="Species", data=iris);
from sklearn.cross_validation import train_test_split# split data

from sklearn import metrics# summarize the classifier
#to split data into training and testing

train, test = train_test_split(iris, test_size = 0.2)

xtrain = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

ytrain = train.Species

xtest = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

ytest = test.Species

from sklearn.linear_model import LogisticRegression

# select the classifier

classifier = LogisticRegression()

# train the classifier

classifier.fit(xtrain, ytrain)

print(classifier)

# make predictions

expected = ytest

predicted = classifier.predict(xtest)

# summarize the fit of the classifier 

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))

# accuracy

print('accuracy is',metrics.accuracy_score(predicted,ytest))
from sklearn.naive_bayes import GaussianNB

# select the classifier

classifier = GaussianNB()

# train the classifier

classifier.fit(xtrain,ytrain)

print(classifier)

# make predictions

expected = ytest

predicted = classifier.predict(xtest)

# summarize the fit of the classifier

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))

# accuracy

print(' accuracy is',metrics.accuracy_score(predicted,ytest))
from sklearn.svm import SVC

# select the classifier

classifier = SVC()

# train the classifier

classifier.fit(xtrain,ytrain)

print(classifier)

# make predictions

expected = ytest

predicted = classifier.predict(xtest)

# summarize the fit of the classifier

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))

# accuracy

print(' accuracy is',metrics.accuracy_score(predicted,ytest))
from sklearn.neighbors import KNeighborsClassifier

# select the classifier

classifier = KNeighborsClassifier(n_neighbors=8)

# train the classifier

classifier.fit(xtrain,ytrain)

print(classifier)

# make predictions

expected = ytest

predicted = classifier.predict(xtest)

# summarize the fit of the classifier

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))

# accuracy

print(' accuracy is',metrics.accuracy_score(predicted,ytest))
from sklearn.tree import DecisionTreeClassifier

# select the classifier

classifier = DecisionTreeClassifier()

# train the classifier

classifier.fit(xtrain,ytrain)

print(classifier)

# make predictions

expected = ytest

predicted = classifier.predict(xtest)

# summarize the fit of the classifier

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))

# accuracy

print(' accuracy is',metrics.accuracy_score(predicted,ytest))