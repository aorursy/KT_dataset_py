# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/UNSW_NB15_training-set.csv")
# View the data as a table, first 10 rows

data.head(10)
#  number of instances (rows) and number attributes (columns) 

data.shape
# Basic Statistics Summary

# describe()- will returns the quick stats such as count, mean, std (standard deviation), 

# min, first quartile, median, third quartile, max on each column of the dataframe

data.describe()
# cov() - Covariance indicates how two variables are related. A positive covariance means 

# the variables are positively related, while a negative covariance means the variables are inversely related. 

# Drawback of covariance is that it does not tell you the degree of positive or negative relation

data.cov()
# corr() - Correlation is another way to determine how two variables are related. 

# In addition to telling you whether variables are positively or inversely related, correlation also 

# tells you the degree to which the variables tend to move together. When you say that two items correlate, 

# you are saying that the change in one item effects a change in another item. You will always talk about correlation 

# as a range between -1 and 1.

data.corr()