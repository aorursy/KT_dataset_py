# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

train_df = pd.read_csv('../input/train.csv', index_col=0)

test_df = pd.read_csv('../input/test.csv', index_col=0)

print('train set and test set :',np.shape(train_df),np.shape(test_df))



#train_df.head()

label = train_df.pop('SalePrice')

print('new train set and test set :',np.shape(train_df),np.shape(test_df))

train_test_set = pd.concat((train_df, test_df))

print(np.shape(train_test_set))

#print train_df.axes

#print train_test_set.axes

train_test_set['MSSubClass'] = train_test_set['MSSubClass'].astype('str')

train_test_set['MSSubClass'].head()
train_test_set['MSSubClass'].value_counts()
train_test_set_dummies = pd.get_dummies(train_test_set)

print( np.shape(train_test_set) , np.shape(train_test_set_dummies))

train_test_set_dummies.head()
train_test_set_dummies.isnull().sum().sort_values(ascending=False).head()

mean_cols = train_test_set_dummies.mean()

#print mean_cols

mean_cols.head(10)

train_test_set_dummies = train_test_set_dummies.fillna(mean_cols)

train_test_set_dummies.isnull().sum().sort_values(ascending=False).head()
numeric_cols = train_test_set.columns[train_test_set.dtypes != 'object']

np.shape(numeric_cols)
numeric_cols_means = train_test_set_dummies.loc[:,numeric_cols].mean()

numeric_cols_std = train_test_set_dummies.loc[:, numeric_cols].std()

train_test_set_dummies.loc[:, numeric_cols] = (train_test_set_dummies.loc[:, numeric_cols] - numeric_cols_means) / numeric_cols_std

train_test_set_dummies.loc[:,numeric_cols].head()
from sklearn.linear_model import Ridge 

from sklearn.linear_model import RidgeCV

from sklearn.model_selection import cross_val_score

train_x = train_test_set_dummies.loc[train_df.index]

test_x = train_test_set_dummies.loc[test_df.index]



alphas = np.logspace(-3, 2, 50)

test_scores = []

clf = Ridge()

#for alpha in alphas:

#    clf = Ridge(alpha)

#    test_score = np.sqrt(-cross_val_score(model, train_x, label, cv=10, scoring='neg_mean_squared_error'))

#    test_scores.append(np.mean(test_score))



clf.fit(train_x,label)

print( clf.score(train_x,label))

predict = clf.predict(test_x)

predict
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': predict})
submission_df.to_csv('output.csv', header=True, index_label='Id')