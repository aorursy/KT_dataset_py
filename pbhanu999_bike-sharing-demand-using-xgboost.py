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
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/train.csv')
train.head()

test.head()

test['casual'] = np.nan
test['registered'] = np.nan
test['count'] = np.nan

test.head()

print(train.info())
print(test.info())

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

print(train.info())

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['DOW'] = train['datetime'].dt.dayofweek
train['hour'] = train['datetime'].dt.hour

train.head()

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['DOW'] = test['datetime'].dt.dayofweek
test['hour'] = test['datetime'].dt.hour

col = ['workingday','temp','year','month','DOW', 'hour']
x = train[col]
y = train['count']

X_test = test[col]
Y_test = test['count']

from sklearn.model_selection import train_test_split
X_train , X_valid ,Y_train , Y_valid = train_test_split(x,y,test_size = 0.25, random_state = 201)

def RSMLE(predictions , realizations):
    predictions_use = predictions.clip(0)
    rmsle = np.sqrt(np.mean(np.array(np.log(predictions_use+1)-np.log(realizations+1))**2))
    return rmsle

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_split=25 , random_state=201)
dtr_model = dtr.fit(X_train,Y_train)

dtr_pred = dtr_model.predict(X_valid)

pd.DataFrame(dtr_model.feature_importances_,index=col)

plt.figure(figsize=(7,7))
plt.scatter(dtr_pred,Y_valid, s=0.2)
plt.xlim(-100,1200)
plt.ylim(-100,1200)
plt.plot([-100,1200],[-100,1200],color = 'pink', linestyle = '-', linewidth =7)
plt.suptitle("Predicted / Actual", fontsize = 20 )
plt.xlabel('pred')
plt.ylabel("y_valid")

RSMLE(dtr_pred,Y_valid)

from sklearn.ensemble import RandomForestRegressor
regress = RandomForestRegressor(n_estimators=500, max_features=4,min_samples_leaf=5, random_state=201)
regress.fit(X_train,Y_train)

predict = regress.predict(X_valid)

plt.figure(figsize=(7,7))
plt.scatter(predict,Y_valid, s=1.9)
plt.xlim(-100,1200)
plt.ylim(-100,1200)
plt.plot([-100,1200],[-100,1200],color = 'purple', linestyle = '-', linewidth =5)
plt.suptitle("Predicted / Actual", fontsize = 20 )
plt.xlabel('pred')
plt.ylabel("y_valid")

RSMLE(predict,Y_valid)

import xgboost as xgb
xgb_train = xgb.DMatrix(X_train,label=Y_train)
xgb_valid = xgb.DMatrix(X_valid)

num_round_for_cv = 500
param = { 'max_depth': 6 , 'eta':0.1 , 'seed' : 201 , 'objective' : 'reg:linear'}

xgb.cv(param,xgb_train,num_round_for_cv,nfold=5,show_stdv=False,verbose_eval=True,as_pandas=False)

num_round = 400
xg_model = xgb.train(param,xgb_train,num_round)
xg_pred = xg_model.predict(xgb_valid)

xg_model.get_fscore()

xgb.plot_importance(xg_model)

plt.figure(figsize=(7,7))
plt.scatter(xg_pred,Y_valid, s=0.6)
plt.xlim(-100,1200)
plt.ylim(-100,1200)
plt.plot([-100,1200],[-100,1200],color = 'orange', linestyle = '-', linewidth =5)
plt.suptitle("Predicted / Actual", fontsize = 20 )
plt.xlabel('xg_pred')
plt.ylabel("y_valid")

RSMLE(xg_pred,Y_valid)

test_dt =  dtr.fit(x,y)
predict_dt = test_dt.predict(X_test)
dt_clipped = pd.Series(predict_dt.clip(0))

test_rf =  regress.fit(x,y)
predict_rt = test_rf.predict(X_test)
rf_clipped = pd.Series(predict_rt.clip(0))

xgbtrain = xgb.DMatrix(x,label=y)
xgbtest = xgb.DMatrix(X_test)
xgmodel = xgb.train(param,xgbtrain,num_round)
xgpred = xgmodel.predict(xgbtest)
xg_clipped = pd.Series(xgpred.clip(0))

