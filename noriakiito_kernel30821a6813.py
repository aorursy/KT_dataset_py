# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_dir = '/kaggle/input/1056lab-student-performance-prediction'

train = pd.read_csv(data_dir + '/train.csv', index_col=0)

test = pd.read_csv(data_dir + '/test.csv', index_col=0)
train
test
train = pd.get_dummies(train, drop_first=True)

train = train.drop(['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], axis=1)
test = pd.get_dummies(test, drop_first=True)

test = test.drop(['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], axis=1)
train_X = train.drop('G3', axis=1)

train_y = train[['G3']]
# 正規分布に従うかどうかのチェック

import scipy.stats as stats

ax = sns.distplot(train_y)

plt.show()

stats.shapiro(train_y)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_features='auto')

rf.fit(train_X, train_y)

print('Training done using Random Forest')



ranking = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=rf.feature_importances_[ranking], y=train_X.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
# use the top 10 features only

train_X = train_X.iloc[:,ranking[:20]]

test = test.iloc[:,ranking[:20]]
import xgboost as xgb

from sklearn.model_selection import GridSearchCV



print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6,8,10,12],

                    'n_estimators': [100,200,250,300]}, verbose=1)

reg_xgb.fit(train_X, train_y)
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor()

reg_lgb = GridSearchCV(lgb_model,

                   {'max_depth': [2,4,6,8,10,12],

                    'n_estimators': [100,200,250,300]},verbose=1)

reg_lgb.fit(train_X, train_y)
print("Parameter optimization")

rf_model = RandomForestRegressor()

reg_rf = GridSearchCV(rf_model,

                   {'max_depth': [2,4,6,8,10,12],

                    'n_estimators': [100,200,250,300]}, verbose=1)

reg_rf.fit(train_X, train_y)
train_X2 = pd.DataFrame( {'XGB': reg_xgb.predict(train_X),

     'LGB': reg_lgb.predict(train_X),

     'RF': reg_rf.predict(train_X)

    })

train_X2
# second-feature modeling using linear regression

from sklearn import linear_model

from sklearn.metrics import mean_absolute_error



reg = linear_model.LinearRegression()

reg.fit(train_X2, train_y)

y_pred = np.round(reg.predict(train_X2))

mean_absolute_error(train_y, np.round(y_pred))
# second-feature modeling using linear regression

from sklearn import linear_model



reg = linear_model.LinearRegression()

reg.fit(train_X2, train_y)



# prediction using the test set

test2 = pd.DataFrame( {'XGB': reg_xgb.predict(test),

     'LGB': reg_lgb.predict(test),

     'RF': reg_rf.predict(test)

    })



# Don't forget to convert the prediction back to non-log scale

y_pred = reg.predict(test2)

y_pred = np.reshape(y_pred,(-1))

# submission

submission = pd.DataFrame({

    'Id': test.index,

    'G3': y_pred

})

submission.to_csv('submit10.csv', index=False)
y_pred