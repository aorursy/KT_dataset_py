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
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score
import os

print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_train.info()
df_train.columns
df_train.F5.unique()
df_train.nunique()
df_test.head()
df_train.head()
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)

df_test.drop(['F1', 'F2'], axis = 1, inplace = True)
categorical_features = ['F3', 'F4', 'F5', 'F7', 'F8', 'F9','F11','F12']

every_column_except_y= [col for col in df_train.columns if col not in ['O/P']]

df_train.describe()
df_train = pd.get_dummies(df_train,columns =categorical_features)

df_test = pd.get_dummies(df_test,columns =categorical_features)
df_test['F3_2'] = 0

df_test['F4_0'] = 0

df_test['F5_1'] = 0

df_test['F5_2'] = 0

df_test['F5_3'] = 0

df_test['F5_4'] = 0

df_test['F5_5'] = 0

df_test['F5_6'] = 0

df_test['F5_7'] = 0

df_test['F12_4'] = 0
columnsnames = ['F6', 'F10', 'F13', 'F14', 'F15', 'F16', 'F17', 'F3_1', 'F3_2',

       'F3_3', 'F3_4', 'F4_0', 'F4_1', 'F5_1', 'F5_2', 'F5_3', 'F5_4', 'F5_5',

       'F5_6', 'F5_7', 'F5_8', 'F5_9', 'F5_10', 'F5_11', 'F5_12', 'F7_0',

       'F7_1', 'F7_2', 'F7_3', 'F7_4', 'F7_5', 'F7_6', 'F7_7', 'F7_8', 'F7_9',

       'F7_10', 'F7_11', 'F7_12', 'F7_13', 'F7_14', 'F7_15', 'F7_16', 'F7_17',

       'F7_18', 'F7_19', 'F7_20', 'F7_21', 'F7_22', 'F7_23', 'F8_0', 'F8_1',

       'F9_0', 'F9_1', 'F9_2', 'F9_3', 'F9_4', 'F9_5', 'F9_6', 'F11_0',

       'F11_1', 'F12_1', 'F12_2', 'F12_3', 'F12_4']

df_test = df_test.reindex(columns=columnsnames)
df_train.drop(['Unnamed: 0'], axis = 1, inplace = True)
df_train
df_test
train_X = df_train.loc[:, 'F3':'F17']

train_y = df_train.loc[:, 'O/P']
every_column_except_y= [col for col in df_train.columns if col not in ['O/P']]

from sklearn.model_selection import train_test_split

X_train,X_val, y_train, y_val = train_test_split(df_train[every_column_except_y],df_train['O/P'],test_size = 0.25, random_state = 33, shuffle = True)
X_train.shape
y_train.shape
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import time

import xgboost as xgb



clf = xgb.XGBRegressor(

    eval_metric = 'rmse',

    nthread = 4,

    eta = 0.1,

    num_boost_round = 80,

    max_depth = 5,

    subsample = 0.5,

    colsample_bytree = 1.0,

    silent = 1,

    )

parameters = {

    'num_boost_round': [10, 25, 50],

    'eta': [0.05, 0.1, 0.3],

    'max_depth': [3, 4, 5],

    'subsample': [0.9, 1.0],

    'colsample_bytree': [0.9, 1.0],

}

    

clf1 = GridSearchCV(clf, parameters, n_jobs=1, cv=2)

clf2 = RandomizedSearchCV(clf, parameters, n_jobs=1, cv=2)



clf1.fit(X_train, y_train)



y_predict1 = clf1.predict(X_val)

print('')



clf2.fit(X_train, y_train)

            

y_predict2 = clf2.predict(X_val)
from sklearn.metrics import mean_squared_error

from math import sqrt



y_predict1 = clf1.predict(X_val)

rms = sqrt(mean_squared_error(y_val, y_predict1))

rms
from sklearn.metrics import mean_squared_error

from math import sqrt



y_predict2 = clf2.predict(X_val)

rms = sqrt(mean_squared_error(y_val, y_predict2))

rms
df_test = df_test.loc[:, 'F3':'F17']

df_test
X_train
df_test.columns
pred = clf1.predict(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result.head()
result.to_csv('output.csv', index=False)
