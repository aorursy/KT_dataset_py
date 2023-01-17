# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import numpy as np # linear algebra

import pandas as pd

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from scipy.stats import boxcox

# data processing, CSV file I/O (e.g. pd.read_csv)

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

ntrain = train.shape[0]

print(train.columns)

ids = test.Id 

target = train.pop('SalePrice')

data = pd.concat([train, test]).reset_index(drop = True)

data.drop(["Id"], axis = 1, inplace = True)

data.dtypes.value_counts().plot(kind = 'bar')

data = data[data.columns[(data.isnull().sum() <1000)]]





cat = data.select_dtypes(include = ['object']).columns

num = data.dtypes[data.dtypes != 'object'].index



categ = pd.get_dummies(data[cat])

all_data = pd.concat([data[num], categ], axis = 1)

all_data = all_data.fillna(all_data.mean())

print(all_data[num].skew())

for c in num:

    if c != 'YearRemodAdd':

        all_data[c],_ = boxcox(all_data[c] + 1)

'''

    s =  StandardScaler()

        s.fit(all_data[c].values)

        all_data[c] = s.transform(all_data[c])

'''   

target = np.log1p(target) 





dtest = xgb.DMatrix(all_data.iloc[ntrain:].values)

print(all_data[num].skew())

dtrain = xgb.DMatrix(all_data.iloc[:ntrain].values, label = target.values)

params = {}

params['objective'] = 'reg:linear'

params['eta'] = 0.1

cv_results = xgb.cv( params, dtrain, num_boost_round = 100, nfold = 4, early_stopping_rounds = 30, seed = 1)



print (cv_results.shape[0])

xgb1 = XGBRegressor(n_estimators = cv_results.shape[0], objective = 'reg:linear', eta = 0.1, maximum_depth = 7, minimum_weight_child = 5, gamma = 0.04, colsample_bytree = 0.65, colsample = 0.7)

paramg = {'subsample': [0.65, 0.7, 0.75], 'colsample_bytree':[0.65, 0.7, 0.75]}

gridsearch = GridSearchCV(estimator = xgb1, param_grid = paramg, scoring = 'neg_mean_squared_error', cv = 4, iid = False)

gridsearch.fit(all_data.iloc[:ntrain ].values, target.values)

print (gridsearch.best_params_, gridsearch.best_score_)

xgb1.fit(all_data.iloc[:ntrain ].values, target.values)

pred = xgb1.predict(all_data.iloc[ntrain:].values)

sub = pd.DataFrame()

sub['Id'] = ids

sub['SalePrice'] = np.exp(pred) - 1

sub.to_csv('submission.csv')

# Any results you write to the current directory are saved as output.

# This Python 3 environment comes with many h