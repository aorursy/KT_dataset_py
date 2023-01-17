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
import numpy as np 
import pandas as pd
from sklearn import *
import lightgbm as lgbm
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

def add_window_feature(df):
    window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]
    for window in window_sizes:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    
    return df

df_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
df_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
sample = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
'''df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('train.csv')
sample=pd.read_csv('sample_submission.csv')'''

df_train = add_window_feature(df_train)
#print(df_train.columns)
df_test = add_window_feature(df_test)
#print(df_test.columns)

X = df_train.drop(['time', 'open_channels'], axis=1)
y = df_train['open_channels']


#model = xgb.XGBRegressor(max_depth=3)
#model = RandomForestRegressor(n_estimators = 1000, random_state = 42,)
#model.fit(X, y)

model = lgbm.LGBMRegressor(n_estimators=1000)
#print("k")
model.fit(X, y)
X_test = df_test.drop(['time'], axis=1)
predicts = model.predict(X_test)

predicts=np.round(np.clip(predicts, 0, 10)).astype(int)
print(len(predicts))

score = accuracy_score(sample.open_channels, predicts)
print(score)

df_test['open_channels']=np.round(np.clip(predicts,0,10).astype(int))
df_test[['time','open_channels']].to_csv("submit.csv",index=False,float_format='%.4f')
len(df_test)