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
#importing test and train data
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
train_data.head(5)
train_data.shape
train_data.store.value_counts()
train_data.item.value_counts()
import matplotlib.pyplot as plt
plt.plot(train_data['sales'])
train_data['date']=pd.to_datetime(train_data['date'])
test_data['date']=pd.to_datetime(test_data['date'])
from datetime import  datetime
train_data['month']=train_data['date'].dt.strftime("%b")
test_data['month']=test_data['date'].dt.strftime("%b")
train_data['weekdays']=train_data['date'].dt.strftime("%a")
test_data['weekdays']=test_data['date'].dt.strftime("%a")
train_data['365days']=train_data['date'].dt.strftime("%j")
test_data['365days']=test_data['date'].dt.strftime("%j")
train_data['days']=train_data['date'].dt.strftime("%d")
test_data['days']=test_data['date'].dt.strftime("%d")
test_data['365days']=test_data['365days'].astype('int64')
test_data['days']=test_data['days'].astype('int64')
train_data['365days']=train_data['365days'].astype('int64')
train_data['days']=train_data['days'].astype('int64')
train_data.info()
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
train_data=pd.get_dummies(train_data)
x=train_data.drop(['date','sales'],axis=1)
y=train_data['sales']
x.head()
model=GradientBoostingRegressor().fit(x,y)
final_pred=model.predict(x)

np.sqrt(mean_squared_error(y,final_pred))
x.info()

x_test=test_data.drop(['date','id'],axis=1)
x_test=pd.get_dummies(x_test)
x_test.info()
x_test['month_Apr']=0
x_test['month_Aug']=0
x_test['month_Dec']=0
x_test['month_Jul']=0
x_test['month_Jun']=0
x_test['month_May']=0
x_test['month_Nov']=0
x_test['month_Oct']=0
x_test['month_Sep']=0
pred=pd.DataFrame(model.predict(x_test),test_data['id'])
pred.rename({0:'sales',},axis=1).to_csv('sample_submission.csv')
