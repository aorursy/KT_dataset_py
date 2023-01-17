# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

dataset = pd.read_csv("../input/heart.csv")
import missingno as msno

msno.matrix(dataset)

dataset.head()
dataset.describe()
dataset.shape
import matplotlib.pyplot as plt

import seaborn as sns



fig, ax = plt.subplots(4,2, figsize = (16,16))

sns.distplot(dataset.age, ax = ax[0, 0])

sns.distplot(dataset.cp, ax = ax[0, 1])

sns.distplot(dataset.trestbps, ax = ax[1, 0])

sns.distplot(dataset.chol, ax = ax[1, 1])

sns.distplot(dataset.thalach, ax = ax[2, 0])

sns.distplot(dataset.oldpeak, ax = ax[2, 1])

sns.distplot(dataset.thal, ax = ax[3, 0])

sns.distplot(dataset.age, ax = ax[3, 1])

# sns.set(style="darkgrid")

# sns.countplot(dataset.sex, palette="Set3")



g = sns.FacetGrid(dataset, col="sex", row="target", margin_titles=True)

g.map(plt.hist, "age")
sufferer_sex = dataset[dataset['target']==1]['sex'].value_counts()

healthy_sex = dataset[dataset['target']==0]['sex'].value_counts()

df = pd.DataFrame([sufferer_sex, healthy_sex])

df.index = ['Sufferer','Healthy']

df.plot(kind='bar',stacked=True, figsize=(9,6), color = ['g', 'c'])
fig = plt.gcf()

fig.set_size_inches(15, 8)

sns.heatmap(dataset.corr(), annot = True, cmap = 'YlGnBu')
# Splitting the dataset into testing set and training set

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier_1 = LogisticRegression()

classifier_1.fit(X_train, y_train)



# Predicting

y_pred = classifier_1.predict(X_test)



# Accuracy

print(classifier_1.score(X_test, y_test))



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualizing

sns.heatmap(cm, annot = True, cmap = 'Greens')
# # K-nearest neighbour

# from sklearn.neighbors import KNeighborsClassifier

# classifier_2 = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)

# classifier_2.fit(X_train, y_train)



# # predicting

# y_pred = classifier_2.predict(X_test)



# # Accuracy

# print(classifier_2.score(X_test, y_test))



# # Confusion Matrix

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred)

# # Visualizing

# sns.heatmap(cm, annot = True, cmap = 'Greens')
# SVM

from sklearn.svm import SVC

classifier_3 = SVC(kernel = 'linear')

classifier_3.fit(X_train, y_train)



# predicting

y_pred = classifier_3.predict(X_test)



# Accuracy

print(classifier_3.score(X_test, y_test))



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualizing

sns.heatmap(cm, annot = True, cmap = 'Greens')
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

classifier_4 = GaussianNB()

classifier_4.fit(X_train, y_train)



# predicting

y_pred = classifier_4.predict(X_test)



# Accuracy

print(classifier_4.score(X_test, y_test))



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualizing

sns.heatmap(cm, annot = True, cmap = 'Greens')
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

classifier_5 = DecisionTreeClassifier(criterion = "entropy", random_state = 0)

classifier_5.fit(X_train, y_train)



# predicting

y_pred = classifier_5.predict(X_test)



# Accuracy

print(classifier_5.score(X_test, y_test))



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualizing

sns.heatmap(cm, annot = True, cmap = 'Greens')
# Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier_6 = RandomForestClassifier(n_estimators = 200, criterion = "entropy", random_state = 0)

classifier_6.fit(X_train, y_train)



# predicting

y_pred = classifier_6.predict(X_test)



# Accuracy

print(classifier_6.score(X_test, y_test))



# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualizing

sns.heatmap(cm, annot = True, cmap = 'Greens')
