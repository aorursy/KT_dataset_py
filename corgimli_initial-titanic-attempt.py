# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
# check data for 

print(train_df.columns[train_df.isnull().any()])

print(test_df.columns[test_df.isnull().any()])