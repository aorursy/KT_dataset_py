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
path_train = '/kaggle/input/boston-housing-dataset/train.csv'

path_test = '/kaggle/input/boston-housing-dataset/test.csv'



df_train = pd.read_csv(path_train)

df_train.head(5)
print(df_train.shape)
# 先觀察未前處理前的平均表現

from sklearn import linear_model



# split to features_data and label

X = df_train.drop(['MEDV'], axis=1)

y = df_train['MEDV']



reg = linear_model.Lasso(alpha=0.1)
from sklearn.model_selection import cross_validate



cv_results = cross_validate(reg, X, y, cv=5, scoring=('neg_mean_squared_error'))

cv_results['test_score'].mean()
corr_table = df_train.corr()

corr_table.style.background_gradient(cmap='ocean') #pandas內建直接可以顯示heatmap
from sklearn.feature_selection import SelectKBest, f_regression

# 排除掉高度共線性的feature """ TAX and RAD"""



selector = SelectKBest(f_regression, k=3)

selector.fit(X, y)

df_name = pd.DataFrame({'col_name':(X.columns).tolist()})

df_value = pd.DataFrame({'p-value':selector.pvalues_})

print(pd.concat([df_name, df_value], axis=1))
# p-value 中 TAX 較 RAD 小, 因此選擇刪除RAD

X = X.drop(['RAD'], axis=1)
# 接著標準化

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

new_X = min_max_scaler.fit_transform(X)
# 改進模型(ensemble)

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import VotingRegressor



# Training classifiers

reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)

reg2 = RandomForestRegressor(random_state=1, n_estimators=10)

reg3 = LinearRegression()

ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])



cv_results = cross_validate(ereg, new_X, y, cv=5, scoring=('neg_mean_squared_error'))

cv_results['test_score'].mean()