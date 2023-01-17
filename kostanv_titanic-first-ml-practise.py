# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')

combine = [test_df, train_df]
train_df.head()
train_df.tail()
print(train_df.columns.values)
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()