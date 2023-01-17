#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train=pd.read_csv("../input/train.csv")

train.head()

test=pd.read_csv("../input/test.csv")

test.head()
train['Survived'].value_counts().plot('bar')

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train = train.drop(columns=['Cabin'])

train.Age=train.Age.fillna(train.Age.mean())

traiu