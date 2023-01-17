#Meghdoot31 - Kernel 01 - Iris Species Dataset

#Import packages that might be useful in this Kernel



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats

from sklearn import preprocessing

from scipy.cluster.hierarchy import dendrogram, linkage



import os
iris = pd.read_csv("../input/Iris.csv")
iris = iris.drop('Id',1)

print(iris.head())

print(iris.info())

print(iris.shape)
matcorr = iris.corr()

mask = np.zeros_like(matcorr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(matcorr, mask=mask, cmap="Purples", vmin=-1, vmax=1, center=0, square=True,annot = True);

plt.show()
sns.pairplot(iris,diag_kind = "kde",hue = "Species",height = 5, vars = ["SepalLengthCm","SepalWidthCm"])
sns.pairplot(iris,hue = "Species",height = 5, vars = ["PetalLengthCm","PetalWidthCm"])
iris["Species"].value_counts().plot(kind = "bar")
## Dimension of the data set



print("The dimension of the Iris dataset are:",iris.shape)

print(" ")

print("The number elements in the Iris dataset is:",iris.size)

print(" ")

print(iris.isnull().sum())

print(" ")

iris = iris.dropna()

print(" ")

print(iris.info())

print(" ")

print("The Unique Species of Iris Flowers and the count are:") 

print(iris["Species"].value_counts())

iris.head()
iris.tail()
iris.describe()
cols = iris.columns

feature1 = cols[0:3]

print(feature1)

label = cols[3]

print(label)

#Shuffle The data

indices = data_norm.index.tolist()

indices = np.array(indices)

np.random.shuffle(indices)
from pandas import get_dummies
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
# One Hot Encode as a dataframe

from sklearn.model_selection import train_test_split

y = get_dummies(y)



# Generate Training and Validation Sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)



# Convert to np arrays so that we can use with TensorFlow

X_train = np.array(X_train).astype(np.float32)

X_test  = np.array(X_test).astype(np.float32)

y_train = np.array(y_train).astype(np.float32)

y_test  = np.array(y_test).astype(np.float32)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# K-Nearest Neighbours

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



Model = KNeighborsClassifier(n_neighbors=8)

Model.fit(X_train, y_train)



y_pred = Model.predict(X_test)
# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test.argmax(0), y_pred.argmax(0)))

print('Accuracy is',accuracy_score(y_pred,y_test))