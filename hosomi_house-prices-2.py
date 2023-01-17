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
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, Imputer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import math

import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
target = df_train['SalePrice']
le = LabelEncoder()
for col in df_train:
    if df_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        col_set = set(list(df_train[col].unique())) | set(list(df_test[col].unique()))
        if len(col_set) <= 2:
            # Train on the training data
            le.fit(df_train[col])
            # Transform both training and testing data
            df_train[col] = le.transform(df_train[col])
            df_test[col] = le.transform(df_test[col])
# one-hot encoding
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)
zscore = lambda x: (x - x.mean()) / x.std()
for col in df_train:
    if col == 'Id':
        continue
    df_train[col] = df_train[col].transform(zscore)
    df_test[col] = df_test[col].transform(zscore)
# Add feature
for col in df_train:
    if len(list(df_train[col].unique())) > 2 and col != 'Id':
        # square
        df_train[col+'_2'] = [x*x for x in df_train[col]]
        df_test[col+'_2'] = [x*x for x in df_test[col]]
        # ln
        df_train[col+'_ln'] = [math.log(x, np.e) if x > 0 else 0 for x in df_train[col]]
        df_test[col+'_ln'] = [math.log(x, np.e) if x > 0 else 0 for x in df_test[col]]
        # exp
        df_train[col+'_exp'] = [math.exp(x) if not np.isnan(x) else np.nan for x in df_train[col]]
        df_test[col+'_exp'] = [math.exp(x) if not np.isnan(x) else np.nan for x in df_test[col]]
        #sin                
        df_train[col+'_sin'] = [math.sin(x) if not np.isnan(x) else np.nan for x in df_train[col]]
        df_test[col+'_sin'] = [math.sin(x) if not np.isnan(x) else np.nan for x in df_test[col]]
reg = xgb.XGBRegressor()
reg_cv = GridSearchCV(reg,
                      {'max_depth': [x for x in range(3,6)],'n_estimators': [x*40 for x in range(6,8)]},
                      verbose=1)
reg_cv.fit(df_train, target)
print(reg_cv.best_params_, reg_cv.best_score_)
reg = xgb.XGBRegressor(**reg_cv.best_params_)
reg.fit(df_train, target)
pred_test = reg.predict(df_test)
submission = df_test[['Id']]
submission['SalePrice'] = list(pred_test)
submission.to_csv('submission2.csv', index=False)