import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import os
os.chdir('../input/')
data=pd.read_csv('earthquake_int.csv')
data.head()
data.shape
data.isna().sum()
data_f=data[["lat","long","depth","xm","md","richter","ms","mb"]]
data_f.dtypes
y=data_f[["richter"]]

x=data_f.drop(["richter"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
x_train.shape
x_test.shape
model=XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)
model.fit(x_train,y_train)
check=model.predict(x_test)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test, check))
rms
t2=pd.Series(check)
t1=pd.Series(y_test.reset_index(drop=True).values[:,0])
Result_comparison=pd.DataFrame({'Actual':t1,'Predicted':t2})
Result_comparison