import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC
# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(5)
test.head(5)
train.info()
train.describe()
pd.isna(train).describe()
train.iloc[:,[0, 1, 2, 5, 6, 7, 9]].apply(lambda x:x.mean())
train.Pclass.value_counts()
len(train.Cabin.unique())
train.Age.mean()
train.Age.unique()
train.groupby('Embarked').size()