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
#Importing python libraries
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model
pd.set_option('display.max_rows',100)# amount of rows that can be seen at a time
# import train and test to play with it
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.info()
test_df.info()
test_df['SalePrice'] = 0
# Adding a row in test data for further calcualtion
train_df.head()
# We are concating to get to know more about the complete dataset
df = pd.concat((train_df,test_df),axis = 0)
df = df.reset_index()
df.info()
df = df.drop(['index'],axis = 1)
df.tail()
df = df.set_index(['Id'])
df.head()
df.describe(include = 'all')
df.isnull().sum()
df.MSZoning.value_counts()
df.LotFrontage.value_counts()
df.to_csv('out.csv')
