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
train = pd.read_csv("../input/cs-challenge/training_set.csv", index_col="ID").drop("MAC_CODE", axis = 1)
train = train.dropna(axis = 1)

from sklearn.preprocessing import StandardScaler



scaler= StandardScaler()

scaler.fit(train)

train = pd.DataFrame(scaler.transform(train), columns = train.columns, index = train.index)

train
from sklearn import linear_model



lasso_reg = linear_model.LassoCV(cv=5, random_state=0, max_iter=10000).fit(train.drop("TARGET",axis=1), train["TARGET"])

print(lasso_reg.score(train.drop("TARGET",axis=1), train["TARGET"]))
import matplotlib.pyplot as plt



plt.figure(figsize=(20,5))

plt.xticks(rotation = 'vertical')

plt.bar(train.drop("TARGET", axis=1).columns, lasso_reg.coef_)
column_list = column_list = train.columns.to_list()

base_cols = [x for x in column_list if x.find('_max') == -1 and x.find('_min') == -1 and x.find('_c') == -1]
train_base = train[base_cols]

train_base
from sklearn import linear_model



lasso_reg_base = linear_model.LassoCV(cv=5, random_state=0, max_iter=20000, fit_intercept=True,normalize = True).fit(train_base.drop("TARGET",axis=1), train_base["TARGET"])

print(lasso_reg_base.score(train_base.drop("TARGET",axis=1), train_base["TARGET"]))
import matplotlib.pyplot as plt



plt.figure(figsize=(20,5))

plt.xticks(rotation = 'vertical')

plt.bar(train_base.drop("TARGET", axis=1).columns, lasso_reg_base.coef_)
test = pd.read_csv("../input/cs-challenge/test_set.csv", index_col = "ID").drop("MAC_CODE",axis=1)

test["TARGET"] = np.ones(len(test.index))

#test[train.columns.to_list()]

test = pd.DataFrame(scaler.transform(test[train.columns.to_list()]), index=test.index, columns=train.columns)

p1 = lasso_reg.predict(test[train.columns.to_list()].drop("TARGET", axis=1))
test['TARGET'] = p1

a1 = pd.DataFrame(scaler.inverse_transform(test[train.columns.to_list()]), index=test.index, columns=test.columns)['TARGET']

a1.to_csv('a1.csv')
p2 = lasso_reg_base.predict(test[train_base.columns.to_list()].drop("TARGET", axis=1))
test["TARGET"] = p2

a2 = pd.DataFrame(scaler.inverse_transform(test[train.columns.to_list()]), index=test.index, columns=test.columns)['TARGET']

a2.to_csv('a2.csv')