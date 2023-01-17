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
train.shape, test.shape
combi = pd.concat([train, test], sort = False, ignore_index=True)
combi.tail()
num_col = [col for col in combi.columns if combi[col].dtype != 'object']

cat_col = [col for col in combi.columns if combi[col].dtype == 'object']
nan_num = [col for col in num_col if combi[col].isnull().any()]

nan_num
combi['LotFrontage'].fillna(combi['LotFrontage'].mean(), inplace = True)

combi['MasVnrArea'].fillna(combi['MasVnrArea'].mean(), inplace = True)
for col in nan_num:

    if combi[col].isnull().any():

        combi[col].fillna(0, inplace = True)
for col in cat_col:

    if combi[col].isnull().any():

        combi[col].fillna('Missing', inplace = True)
combi[num_col].head()
for col in num_col:

    combi[col] = np.log1p(combi[col])

       

combi.head()
train_hot = combi.iloc[:train.shape[0], 1:-1]

y = combi.iloc[:train.shape[0], -1]

train_hot.shape, y.shape

test_hot = combi.iloc[train.shape[0] : , 1:-1]
from sklearn.preprocessing import OneHotEncoder



myOneHot = OneHotEncoder(handle_unknown= 'ignore', sparse=False)



train_OneHot_cat_col = pd.DataFrame(myOneHot.fit_transform(train_hot[cat_col]))

test_OneHot_cat_col = pd.DataFrame(myOneHot.transform(test_hot[cat_col]))



# add the index back

train_OneHot_cat_col.index = train_hot.index

test_OneHot_cat_col.index = test_hot.index



#remove the object columns 

train_hot.drop(cat_col, axis = 1, inplace = True)

test_hot.drop(cat_col, axis = 1, inplace = True)



#add the onehot columns to the train, valid and test

train_hot_encoded = pd.concat([train_hot, train_OneHot_cat_col], axis = 1)

test_hot_encoded = pd.concat([test_hot, test_OneHot_cat_col], axis = 1)
train_hot_encoded.shape, test_hot_encoded.shape
import pandas as pd

from statistics import *

from math import sqrt

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression,Ridge, Lasso

from sklearn.model_selection import cross_val_score

import matplotlib

# %pylab inline
train_hot_encoded.head()
from sklearn.linear_model import LinearRegression, Ridge, Lasso



lin_reg = LinearRegression()



score = mean(np.sqrt((-cross_val_score(lin_reg, train_hot_encoded, y, scoring='neg_mean_squared_error', cv=10))))

print(score)





# score = mean(sqrt(-cross_val_score(lin_reg, train_hot_encoded, y, scoring='neg_mean_squared_error', cv=10)))

# print(score)

alphas = np.logspace(-5, 2, 20)

scores = []



for i in alphas:

    model_ridge = Ridge(alpha = i)

    score = mean(np.sqrt(-cross_val_score(model_ridge, train_hot_encoded, y, scoring='neg_mean_squared_error', cv=10 )))

    scores.append(score)

    

df = pd.DataFrame(list(zip(alphas, scores)), columns = ['alphas', 'scores'])

min_score = df['scores'].idxmin()

df.iloc[min_score, :]    
alphas = np.logspace(-5, 2, 20)

scores_lass = []



for i in alphas:

    model_lass = Lasso(alpha = i)

    score = mean(np.sqrt(-cross_val_score(model_lass, train_hot_encoded, y, scoring='neg_mean_squared_error', cv=10 )))

    scores_lass.append(score)

    

min(scores_lass)

df_lasso = pd.DataFrame(list(zip(alphas, scores_lass)), columns = ['alphas', 'scores_lass'])

min_score = df_lasso['scores_lass'].idxmin()

df_lasso.iloc[min_score, :][0]
model_final = Lasso(alpha = df_lasso.iloc[min_score, :][0])

model_final.fit(train_hot_encoded, y)

prediction = model_final.predict(test_hot_encoded)

converted_prediction = np.expm1(prediction)
submission2 = pd.DataFrame({"Id" : test['Id'],

                           'SalePrice': converted_prediction}, index = None)



submission2.to_csv('Submission2.csv')