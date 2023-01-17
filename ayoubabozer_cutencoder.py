# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
train.dropna(inplace=True)

test.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
train['Age_Range'] = pd.cut(train['Age'], bins=4)

sns.countplot(train['Age_Range'])
test['Age_Range'] = pd.cut(test['Age'], bins=4)

sns.countplot(test['Age_Range'])
cutted = pd.cut(pd.concat([train['Age'], test['Age']]), 4, retbins=True)

intervals = cutted[1]
sns.countplot(pd.cut(train['Age'], intervals))
sns.countplot(pd.cut(test['Age'], intervals))
cutted = pd.cut(pd.concat([train['Age'], test['Age']]), 4, retbins=True)

intervals = cutted[1]

train['Age_Range'] = pd.cut(train['Age'], intervals)

test['Age_Range'] = pd.cut(test['Age'], intervals)

age_encoder = LabelEncoder().fit(pd.concat([train['Age_Range'], test['Age_Range']]))

train['Age_Range'] = age_encoder.transform(train['Age_Range'])

test['Age_Range'] = age_encoder.transform(test['Age_Range'])
sns.countplot(train['Age_Range'])
sns.countplot(test['Age_Range'])
class CutEncoder():

    """Encode numeric values with equal interval cut value between 0 and n_classes-1."""

    def __init__(self):

        self.intervals = None

    def fit(self, x, bins):

        self.x = x

        self.bins = bins

        cutted = pd.cut(self.x, self.bins, retbins=True)

        self.intervals = cutted[1]

    def transform(self, y):

        return pd.cut(y, self.intervals, labels=list(range(len(self.intervals)-1)))

        
cut_encoder = CutEncoder()

cut_encoder.fit(pd.concat([train['Age'], test['Age']]), 4)

train['Age_Range'] = cut_encoder.transform(train['Age'])

test['Age_Range'] = cut_encoder.transform(test['Age'])
sns.countplot(train['Age_Range'])
sns.countplot(test['Age_Range'])