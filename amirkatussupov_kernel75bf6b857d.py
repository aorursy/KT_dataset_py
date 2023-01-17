# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample_submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
from __future__ import print_function, division



%matplotlib inline



import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import graphviz 

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random

from sklearn.model_selection import train_test_split

import seaborn as sns

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn import tree

from sklearn.svm import SVC

from sklearn.datasets import load_iris

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn import ensemble

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree.export import export_text
train
test
train.columns
train.info()
train.isna().sum()
train['SalePrice'].describe()
sns.distplot(train['SalePrice']);

print (train.SalePrice.skew())
plt.scatter(x= 'GrLivArea', y='SalePrice', data = train)
sns.lmplot(x='GarageArea',y='SalePrice',data=train)
correlation = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(correlation, vmax=.8, square=True);
cols = correlation.nlargest(20, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.75)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.line()
train.shape
train['PoolQC'].isnull().sum(),train['MiscFeature'].isnull().sum(),train['Alley'].isnull().sum()
train.drop(['Alley'],axis=1,inplace=True)

train.drop(['MiscFeature'],axis =1,inplace=True)

train.drop(['PoolQC'],axis=1,inplace=True)

train.drop(['FireplaceQu'],axis=1,inplace=True)

train.drop(['Fence'],axis=1,inplace=True)

train.drop(['GarageYrBlt'],axis=1,inplace=True)
train.shape
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())

train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])


X_train = train.drop('SalePrice', axis=1)

y_train = train.SalePrice

X_test = test
dum_train_X = pd.get_dummies(X_train)
dum_test_X = pd.get_dummies(X_test)
X_train, X_test = dum_train_X.align(dum_test_X, join='left', axis=1)
si = SimpleImputer()
X_train = si.fit_transform(X_train)
X_test = si.transform(X_test)
lg = LinearRegression()

lg.fit(X_train, y_train)

lg_preds = lg.predict(X_test)
submission= pd.DataFrame({'Id': test.Id, 'SalePrice':lg_preds})
submission.to_csv('submission.csv', index=False)