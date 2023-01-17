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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn import svm

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
df_train.describe().transpose()
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
corr = df_train.corr()
corr.style.background_gradient(cmap='coolwarm')
sns.pairplot(df_train[['F10','F6','O/P']], diag_kind='kde')

df_train.drop(['F10','F6'], axis = 1, inplace = True)

df_train.isnull().any()
X = df_train.loc[:, 'F3':'F17']
y = df_train.loc[:, 'O/P']
from sklearn.model_selection import train_test_split
train_X, dev_X, train_y, dev_y = train_test_split(X, y, test_size = 0.40,shuffle=True)
rf = RandomForestRegressor(n_estimators=250, random_state=40,oob_score=True)
rf.fit(train_X, train_y)
train_pred=rf.predict(X)
mean_squared_error(y,train_pred)
rfdevpred=rf.predict(dev_X)
mean_squared_error(dev_y,rfdevpred)
rf.fit(X,y)
train_pred=rf.predict(X)
mean_squared_error(y,train_pred)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.09,
                          alpha = 5, n_estimators = 200, max_depth=8,
                          base_score=0.4, gamma=10, subsample=0.70,
                          num_parallel_tree=10, random_state=40,
                          grow_policy='lossguide',max_leaves=15,
                          )
xg_reg.fit(train_X,train_y)
xgb_train_pred=xg_reg.predict(X)
mean_squared_error(y,xgb_train_pred)
xgbdevpred=xg_reg.predict(dev_X)
mean_squared_error(dev_y,xgbdevpred)
xg_reg.fit(X,y)
xgb_train_pred=xg_reg.predict(X)
mean_squared_error(y,xgb_train_pred)
sv_reg=svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1,degree=3)
sv_reg.fit(train_X,train_y)
sv_train_pred=sv_reg.predict(X)
mean_squared_error(y,sv_train_pred)
svdevpred=sv_reg.predict(dev_X)
mean_squared_error(dev_y,svdevpred)
sv_reg.fit(X,y)
sv_train_pred=sv_reg.predict(X)
mean_squared_error(y,xgb_train_pred)
df_test.drop(['F1','F2','F6','F10'], axis = 1, inplace = True)
df_test = df_test.loc[:, 'F3':'F17']
xg_pred=xg_reg.predict(df_test)
print(xg_pred)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(xg_pred)
result.head()
result.to_csv('output.csv', index=False)
