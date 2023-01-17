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
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import indent
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
pd.set_option('max_rows', 12)
pd.set_option('max_columns', 100)
pd.options.display.float_format = '{0:.2f}'.format
data = pd.read_csv('../input/train.csv')
data
# dealing with missing data

col_to_drop = data.columns[data.notna().sum() < 1300]
col_to_drop = col_to_drop.tolist() + ['GarageQual', 'GarageCond']
# columns from values counting below
col_to_drop = col_to_drop + (['Street', 'LandContour', 'Utilities', 'LandSlope', 'BsmtCond', 'Heating', 'CentralAir', 'Electrical', 'BsmtHalfBath', 'KitchenAbvGr', 'Functional', 'GarageQual', 'GarageCond', 'PavedDrive'])
data = data.drop(columns=col_to_drop)
data = data.dropna()
data.dtypes.value_counts()
types = data.dtypes == 'object'
col_label = types.index[types]
col_number = types.index[types == False]
def diffrent_values(col, treshhold):
    len_val = len(data[col].value_counts(col))
    if len_val < treshhold:
        print("{} wartości kolumny {}:".format(len_val, col))
        print(indent(data[col].value_counts().to_string(), "    "))
        print()
for col in data.columns:
    diffrent_values(col, 8)
X_hot = pd.get_dummies(data)
# praparing data for model fitting
y = data.SalePrice
data = data.drop('SalePrice', axis=1)

train_X, test_X, train_y, test_y = train_test_split(X_hot, y, random_state = 0)
# preparing submission
test_data = pd.read_csv('../input/test.csv')
test_data = test_data.drop(columns=col_to_drop)
test_data = pd.get_dummies(test_data)
final_subm_X = test_data.drop('Id', axis=1)
final_train_X, final_subm_X = train_X.align(final_subm_X, join='inner', axis=1)
final_test_X, _ = test_X.align(final_subm_X, join='inner', axis=1)
# err_list = []
# values = np.arange(0.1, 0.2, 0.005)
# for n in values:
#     model1 = XGBRegressor(n_estimators=10000, learning_rate= n)  # 0.065
#     model1.fit(X=final_train_X, y=train_y, early_stopping_rounds=40, eval_set=[(final_test_X, test_y)], verbose=False)
#     pred1 = model1.predict(final_test_X)
#     err_list.append(mean_absolute_error(pred1, test_y))
#     print(f"{n:.4f}: {int(mean_absolute_error(pred1, test_y))}")
# plt.plot(values, err_list)
# err_list = []
# values = np.arange(30, 60, 2)
# for n in values:
#     model1 = XGBRegressor(n_estimators=10000, learning_rate= 0.13)  # 0.065
#     model1.fit(X=final_train_X, y=train_y, early_stopping_rounds=n, eval_set=[(final_test_X, test_y)], verbose=False)
#     pred1 = model1.predict(final_test_X)
#     err_list.append(mean_absolute_error(pred1, test_y))
#     print(f"{n:.4f}: {int(mean_absolute_error(pred1, test_y))}")
# plt.plot(values, err_list)
# models

model1 = XGBRegressor(n_estimators=10000, learning_rate=0.14)
model1.fit(X=final_train_X, y=train_y, early_stopping_rounds=40, eval_set=[(final_test_X, test_y)])
pred1 = model1.predict(final_test_X)
print(model1)
print("Mean Absolute Error : " + str(mean_absolute_error(pred1, test_y)))
pred1
mod2 = RandomForestRegressor()
mod2.fit(X=final_train_X, y=train_y)
pred2 = mod2.predict(final_test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(pred2, test_y)))
mod3 = DecisionTreeRegressor()
mod3.fit(X=final_train_X, y=train_y)
pred3 = mod3.predict(final_test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(pred3, test_y)))
subm_pred = model1.predict(final_subm_X)
subm_pred
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': subm_pred})
my_submission.to_csv('submission3.csv', index=False)
sns.kdeplot(my_submission.SalePrice)