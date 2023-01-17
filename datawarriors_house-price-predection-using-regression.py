import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime



from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score

from sklearn.metrics import classification_report , confusion_matrix





from sklearn.linear_model import LogisticRegression





from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import os

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv' , index_col= 'Id')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv'  , index_col= 'Id')

train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)

label = train[['SalePrice']]

train.drop('SalePrice' , axis = 1 , inplace=True)

train.head(3)
label.head()
train.info()
numerical_col = []

cat_col = []

for x in train.columns:

    if train[x].dtype == 'object':

        cat_col.append(x)

        print(x+': ' + str(len(train[x].unique())))

    else:

        numerical_col.append(x)

        

print('CAT col \n', cat_col)

print('Numerical col\n')

print(numerical_col)
numerical_col.remove('MSSubClass')

cat_col.append('MSSubClass')
train_num = train[numerical_col]

train_num.head()
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

#imputer = Imputer(missing_values='NaN' , strategy='median' , axis = 0)

imputer = imputer.fit(train_num)

train_num = imputer.transform(train_num)
test_num = imputer.transform(test[numerical_col])
print(train_num.shape)

print(test_num.shape)
X_train , X_test , y_train , y_test=  train_test_split(train_num , label , test_size= 0.2 , random_state=123)
from sklearn.linear_model import LinearRegression

clf = LinearRegression(normalize=True)

scores = cross_val_score(clf, X_train, y_train, cv=5).mean()

scores
from sklearn.linear_model import Lasso

clf = Lasso(alpha=0.3, normalize=True)

scores = cross_val_score(clf, X_train, y_train, cv=5).mean()

scores
from sklearn.linear_model import ElasticNet

clf = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)

scores = cross_val_score(clf, X_train, y_train, cv=5).mean()

scores
import xgboost

clf=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

colsample_bytree=1, max_depth=7)

scores = cross_val_score(clf, X_train, y_train, cv=5).mean()

scores
train_cat = train[cat_col]

test_cat = test[cat_col]

print(train_cat.info())

print(test_cat.info())
dropp = ['MiscFeature' , 'PoolQC' , 'Fence' ,'Alley' ]

train_cat.drop(columns=dropp , axis=1, inplace=True)
train_cat = train_cat.astype('category')

print(train_cat.info())
test_cat.drop(columns=dropp , axis=1, inplace=True)

test_cat = test_cat.astype('category')

test_cat.info()
most_freq = {}

for col in train_cat.columns:

    p = train_cat[col].mode()[0] 

    train_cat[col].fillna(p, inplace=True)

    most_freq[col] = p
for col in train_cat.columns:

    test_cat[col].fillna(most_freq[col], inplace=True)
print(test_cat.info())

print(train_cat.info())
train_cat.head(2)
test_cat.head(2)
train_num =pd.DataFrame(train_num)

train_num.head(2)
test_num =pd.DataFrame(test_num)

test_num.head(2)
for col in train_cat:

    train_cat[col] = train_cat[col].cat.codes

for col in test_cat:

    test_cat[col] = test_cat[col].cat.codes
train_cat.head(2)
train_num.index = train_cat.index
test_num.index = test_cat.index
train_cat = pd.get_dummies(train_cat)

test_cat = pd.get_dummies(test_cat)
train_ = train_num.join(train_cat)
test_ = test_num.join(test_cat)
scalar = MinMaxScaler()

train_ = scalar.fit_transform(train_)

test_ = scalar.transform(test_)
# import xgboost

# clf=xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.07, gamma=0, subsample=0.75,

# colsample_bytree=1, max_depth=7)

# scores = cross_val_score(clf, train_, label, cv=5).mean()

# scores
import lightgbm as lgb

lightgbm = lgb.LGBMRegressor(objective='regression', 

                                       num_leaves=8,

                                       learning_rate=0.03, 

                                       n_estimators=4000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )

scores = cross_val_score(lightgbm, train_, label, cv=5).mean()

scores
clf.fit(train_ , label)

pre = clf.predict(test_)

submit = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submit.head()





submit.SalePrice = pre



submit.to_csv('submit.csv', index = False)
submit.head()