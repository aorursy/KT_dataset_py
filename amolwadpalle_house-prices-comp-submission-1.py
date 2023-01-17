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
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use('fivethirtyeight')

from scipy import stats
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

train.head()

columns = train.columns
test1 = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
train.drop('Id',axis = 1, inplace = True)

test.drop('Id',axis = 1, inplace = True)

train.head()
train.info()
num_col = [col for col in train.columns if train[col].dtype != 'object']

cat_col = [col for col in train.columns if train[col].dtype == 'object']
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(), inplace = True)

test['MasVnrArea'].fillna(test['MasVnrArea'].mean(), inplace = True)
train['GarageYrBlt'].fillna(train['YearBuilt'], inplace = True)

test['GarageYrBlt'].fillna(test['YearBuilt'], inplace = True)
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace = True)

test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace = True)
train[cat_col].isnull().sum().sort_values(ascending = False)
for col in cat_col:

    train[col].fillna('None', inplace = True)

    test[col].fillna('None', inplace = True)
from sklearn.preprocessing import OneHotEncoder



myOneHot = OneHotEncoder(handle_unknown= 'ignore', sparse=False)



train_X_OneHot = pd.DataFrame(myOneHot.fit_transform(train[cat_col]))

test_X_OneHot = pd.DataFrame(myOneHot.transform(test[cat_col]))



# add the index back

train_X_OneHot.index = train.index

test_X_OneHot.index = test.index



#remove the object columns 

train.drop(cat_col, axis = 1, inplace = True)

test.drop(cat_col, axis = 1, inplace = True)



#add the onehot columns to the train, valid and test

train_hot = pd.concat([train, train_X_OneHot], axis = 1)

test_hot = pd.concat([test, test_X_OneHot], axis = 1)





y = train['SalePrice']

train_hot.drop('SalePrice', axis = 1, inplace = True)
test_hot.shape, train_hot.shape
import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression,Ridge, Lasso

from sklearn.model_selection import cross_val_score

import matplotlib

# %pylab inline
from sklearn.linear_model import LinearRegression, Ridge, Lasso



lin_reg = LinearRegression()



score = mean(sqrt(-cross_val_score(lin_reg, train_hot, y, scoring='neg_mean_squared_error', cv=10)))

print(score)

alphas = np.logspace(-5, 2, 20)

scores = []



for i in alphas:

    model_ridge = Ridge(alpha = i)

    score = mean(sqrt(-cross_val_score(model_ridge, train_hot, y, scoring='neg_mean_squared_error', cv=10 )))

    scores.append(score)

    

df = pd.DataFrame(list(zip(alphas, scores)), columns = ['alphas', 'scores'])

min_score = df['scores'].idxmin()

df.iloc[min_score, :]    
alphas = np.logspace(-5, 2, 20)

scores_lass = []



for i in alphas:

    model_lass = Lasso(alpha = i)

    score = mean(sqrt(-cross_val_score(model_ridge, train_hot, y, scoring='neg_mean_squared_error', cv=10 )))

    scores_lass.append(score)

    

min(scores_lass)
nan_cols = [col for col in test_hot.columns if test_hot[col].isnull().any()]

nan_cols
for col in nan_cols:

    test_hot[col].fillna(0, inplace = True)
model_final = Ridge(alpha = 18.329807)

model_final.fit(train_hot, y)

prediction = model_final.predict(test_hot)
test.head()
test_submission = pd.DataFrame({'Id' : test1['Id'],

                                'SalePrice' : prediction})

test_submission.to_csv('test_submission1.csv')