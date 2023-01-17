import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.info()
train_df.describe(include=['O'])
print(train_df.columns.values)

K=list(train_df.X1.unique())

train_df.dtypes
train_df['X0'] = train_df['X0'].astype('category')

train_df['X1'] = train_df['X1'].astype('str').astype('category')

train_df['X2'] = train_df['X2'].astype('category')

train_df['X3'] = train_df['X3'].astype('category')

train_df['X4'] = train_df['X4'].astype('category')

train_df['X5'] = train_df['X5'].astype('category')

train_df['X6'] = train_df['X6'].astype('category')

train_df['X8'] = train_df['X8'].astype('category')



cat_columns = train_df.select_dtypes(['category']).columns

train_df[cat_columns] = train_df[cat_columns].apply(lambda x: x.cat.codes)

train_df.head()
train_df.dtypes
from sklearn.linear_model import LinearRegression

X=train_df.drop(['ID','y'],axis=1)

lr=LinearRegression()

lr.fit(X,train_df.y)

X_test=test_df.drop('ID',axis=1)

mse=np.mean((lr.predict(X)-train_df.y))**2

print(mse)

l=pd.DataFrame(lr.predict(X))





l.to_csv('foo.csv')