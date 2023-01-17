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
data=pd.read_csv('../input/Daily_Demand_Forecasting_Orders.csv')
data.head()
data.groupby([data.columns[0],data.columns[1]]).mean()
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X=data[data.columns[:12]]
y=data[data.columns[12]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
r_squared=[]
y_pred=[]
models_list=[LinearRegression(),Ridge(alpha=1.0)]
for model in models_list:
    pipeline=Pipeline([('scaler',StandardScaler()),('lr',model)])
    pipeline.fit(X_train,y_train)
    print('Training set r_squared: ',pipeline.score(X_train,y_train))
    r_squared.append(pipeline.score(X_test,y_test))
    y_pred.append(pipeline.predict(X_test))
r_squared
week_day_test=range(1,X_test.shape[0]+1)
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(week_day_test,y_pred[0],'b')#Linear Regression
plt.plot(week_day_test,y_test,'r')#Test_data
plt.plot(week_day_test,y_pred[1],'g')#Ridge Regression
plt.show()