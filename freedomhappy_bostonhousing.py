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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor





#显示所有列

pd.set_option('display.max_columns', None)

#显示所有行

pd.set_option('display.max_rows', None)

#设置value的显示长度为100，默认为50

pd.set_option('max_colwidth',100)
boston_housing = pd.read_csv('../input/housing.csv')

boston_housing.head()
def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

train_data,test_data = split_train_test(boston_housing,0.2)

features = train_data.columns.drop('MEDV')

train_X  = train_data[features]

train_y  = train_data['MEDV']
def display_scores(scores):

    print("Scores:",scores)

    print("Mean:",scores.mean())

    print("Standard deviation:",scores.std())
linerReg = LinearRegression()

scores = -cross_val_score(linerReg,train_X,train_y,scoring = "neg_mean_absolute_error",cv=10)

display_scores(scores)
decisionTreeReg = DecisionTreeRegressor(random_state=0)

scores = -cross_val_score(decisionTreeReg,train_X,train_y,scoring = "neg_mean_absolute_error",cv=10)

display_scores(scores)
randomForestReg = RandomForestRegressor(random_state=0)

scores = -cross_val_score(randomForestReg,train_X,train_y,scoring = "neg_mean_absolute_error",cv=10)

display_scores(scores)
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_predict

l_pre = cross_val_predict(linerReg, train_X, train_y, cv=10)

d_pre = cross_val_predict(decisionTreeReg, train_X, train_y, cv=10)

r_pre = cross_val_predict(randomForestReg, train_X, train_y, cv=10)
train_y = train_y // 5

l_pre = l_pre // 5

d_pre = d_pre // 5

r_pre = r_pre // 5
l_score = f1_score(train_y,l_pre,average='micro')

d_score = f1_score(train_y,d_pre,average='micro')

r_score = f1_score(train_y,r_pre,average='micro')

print('linearReg score:',l_score,'\ndecision tree score:',d_score,'\nrandom forest score:',r_score)