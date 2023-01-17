import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import datasets

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/train.csv')

data.head()
data_null = data.isnull().sum().sort_values(ascending=False)

data_null[data_null>0]
data.fillna(0,inplace=True)

data.head()
sns.distplot(data['SalePrice'])
categoricals= []

for col,col_type in data.dtypes.iteritems():

    if col_type =='O':

        categoricals.append(col)

    else:

        data[col].fillna(0,inplace=True)
data=pd.get_dummies(data,columns=categoricals,dummy_na=True)

data.head()
dependent_variable= "SalePrice"

x= data[data.columns.difference([dependent_variable])]

y = data[dependent_variable]

lr = LogisticRegression()

lr.fit(x,y)
lr.score(x,y)