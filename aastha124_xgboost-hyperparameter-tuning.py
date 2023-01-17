import pandas as pd
import numpy as np
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")
train.head()
train.describe()
test=pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/test.csv")
test.head()
#Converting date to datetime format
train['date']=pd.to_datetime(train['date'])
test['date']=pd.to_datetime(test['date'])

#Extracting dayofweek,dayofyear,year,month for training set

train['weekday']=train['date'].dt.dayofweek
train['dayofyear']=train['date'].dt.dayofyear
train['year']=train['date'].dt.year
train['month']=train['date'].dt.month

#Extracting dayofweek,dayofyear,year,month for testing set

test['weekday']=test['date'].dt.dayofweek
test['dayofyear']=test['date'].dt.dayofyear
test['year']=test['date'].dt.year
test['month']=test['date'].dt.month
import plotly.express as px
px.box(x=train['year'],y=train['sales'],title="Yearly Sales")
px.line(x=train['date'],y=train['sales'],title="Daily Sales")
px.box(x=train['month'],y=train['sales'],title="Monthly Sales")
train.isnull().sum()
train.skew()
train.columns
X=train.copy()
X.drop(['sales','date'],axis=1,inplace=True)
y=train['sales']

from sklearn.model_selection import train_test_split

#Splitting data into training and validation test
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=123)

print("Shape of training features:",X_train.shape)
print("Shape of training labels:",y_train.shape)
print("Shape of validation features:",X_val.shape)
print("Shape of validation labels:",y_val.shape)
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
xgb=XGBRegressor(random_state=123)

XBG_score=cross_val_score(xgb,X_train,y_train,cv=5,scoring='neg_mean_squared_error',verbose=15)
print("MSE:",-XBG_score.mean())
print("RMSE:",np.sqrt(-XBG_score.mean()))
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
xgb.fit(X_train,y_train)
training_predictions=xgb.predict(X_train)
print("SMAPE score:",smape(y_train,training_predictions))
xgb
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'min_child_weight': np.arange(0.0001, 0.5, 0.001),
    'gamma': np.arange(0.0,40.0,0.005),
    'learning_rate': np.arange(0.0005,0.3,0.0005),
    'subsample': np.arange(0.01,1.0,0.01),}
#Bayesian optimization over hyper parameters.

from skopt import BayesSearchCV
tuned_XGB=BayesSearchCV(xgb,param_grid,cv=3,scoring='neg_mean_squared_error',random_state=123,verbose=15)
tuned_XGB.fit(X_train,y_train)
"""
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
"""
tuned_XGB.best_estimator_
training_predictions=tuned_XGB.predict(X_train)
print("SMAPE score:",smape(y_train,training_predictions))
validation_predictions=tuned_XGB.predict(X_val)
print("SMAPE score:",smape(y_val,validation_predictions))
X=test.copy()
X.drop(['id','date'],axis=1,inplace=True)

test_predictions=tuned_XGB.predict(X)
final_test=pd.DataFrame()
final_test['id']=test['id']
final_test['sales']=test_predictions
print(final_test.head())

final_test.to_csv("submission.csv", index=False)
