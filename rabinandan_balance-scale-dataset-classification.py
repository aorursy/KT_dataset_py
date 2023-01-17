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
#load dataset

input_data =  pd.read_csv("../input/balance-scale.csv")
input_data.head(5)
input_data.info()
input_data.mean()
##check if null

null = input_data.isnull().sum()

print(null)

##desribe data

input_data.describe()
import seaborn



seaborn.distplot(input_data['left-weight'])
#box plot : https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/

import matplotlib.pyplot as plt

input_data.boxplot()
input_data.hist()
##to show co-relations

from pandas.plotting import scatter_matrix

scatter_matrix(input_data, alpha=0.2, figsize=(6, 6), diagonal='kde')
input_labels = input_data['%class name']

input_features = input_data.drop(['%class name'], axis = 1)
#create unique ids for different class in lebels

input_labels, string_labels = pd.factorize(input_labels)
print(input_labels)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(input_features, input_labels, test_size = 0.2)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
##First apply some classification algorithm 

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)

clf.predict(X_test)

clf.score(X_test, y_test)
##now predict on same input data

logistic_output = clf.predict(X_train) 

print(logistic_output)
#now apply some clustering algorithm

from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters = 3).fit(X_train)

agglomerative_output = clustering.labels_

print(agglomerative_output)
from sklearn.metrics import confusion_matrix

##confusion_matrix(y_true, y_pred) 

##if i assume the logistic output is correct and agglomerative output is predicted then

confusion_matrix(logistic_output, agglomerative_output)