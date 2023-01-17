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
train_df = pd.read_csv('../input/train.csv',index_col=0)

test_df = pd.read_csv('../input/test.csv',index_col=0)
train_df.head()
train_df.shape
prices = pd.DataFrame({"price":train_df["SalePrice"],"log(price+1)":np.log1p(train_df["SalePrice"])})

prices.hist()
prices.describe()
y_train = np.log1p(train_df.pop('SalePrice'))

y_train
all_df = pd.concat((train_df,test_df),axis=0)

all_df.shape
all_df.MSSubClass.dtype
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
all_df['MSSubClass'].value_counts()
#生成one_hot 编码

pd.get_dummies(all_df['MSSubClass'],prefix='MSSubClass').head()
all_dummy_df = pd.get_dummies(all_df)

all_dummy_df.shape
# 查看缺失数据

all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)
# 查看平均值

mean_cols = all_dummy_df.mean()

mean_cols.head(10)
# 用平均值填补

all_dummy_df = all_dummy_df.fillna(mean_cols)
all_dummy_df.isnull().sum().sum()
# 标准化数值数据 提取数值字段

numeric_cols = all_df.columns[all_df.dtypes != 'object']

numeric_cols
# 计算均值，标准差，减去均值/标准差

numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()

numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()

all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols]-numeric_col_means)/numeric_col_std
dummy_train_df = all_dummy_df.loc[train_df.index]

dummy_test_df = all_dummy_df.loc[test_df.index]
dummy_train_df.shape,dummy_test_df.shape
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
X_train = dummy_train_df.values

X_test = dummy_test_df.values
alphas = np.logspace(-3,2,50)

test_scores = []

for alpha in alphas:

    clf = Ridge(alpha)

    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring = 'neg_mean_squared_error'))

    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt

plt.plot(alphas,test_scores)

plt.title("Alpha as cv Error")
from sklearn.ensemble import RandomForestRegressor
max_features = [.1,.3,.5,.7,.9,.99]

test_scores = []

for max_feat in max_features:

    clf = RandomForestRegressor(n_estimators=200,max_features=max_feat)

    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv=5,scoring='neg_mean_squared_error'))

    test_scores.append(np.mean(test_score))
plt.plot(max_features,test_scores)

plt.title("Max Features vs CV Error")
ridge = Ridge(alpha=15)

rf = RandomForestRegressor(n_estimators=500,max_features=.3)
ridge.fit(X_train,y_train)

rf.fit(X_train,y_train)
y_ridge = np.expm1(ridge.predict(X_test))

y_rf = np.expm1(rf.predict(X_test))
y_final=(y_ridge+y_rf)/2

y_final
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_final})
submission_df.to_csv('submission.csv',index=False)