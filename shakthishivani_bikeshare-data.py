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

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
bike_data = pd.read_csv("../input/bike_share.csv")
bike_data.info()
bike_data.head()
bike_data.corr()
sns.pairplot(data=bike_data[["season","workingday","temp","atemp","windspeed","count"]])
bike_data.season.value_counts().plot(kind="pie")
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
y = bike_data["count"]

x = bike_data.drop(columns=["season","workingday","holiday","weather","humidity","casual","registered","count"])
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.3,random_state = 42)
model=LinearRegression()

model.fit(train_x,train_y)
model.coef_
model.intercept_
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("MSE - Train :" ,mean_squared_error(train_y,train_predict))

print("MSE - Test :" ,mean_squared_error(test_y,test_predict))

print("MAE - Train :" ,mean_absolute_error(train_y,train_predict))

print("MAE - Test :" ,mean_absolute_error(test_y,test_predict))

print("R2 - Train :" ,r2_score(train_y,train_predict))

print("R2 - Test :" ,r2_score(test_y,test_predict))

print("Mape - Train:" , np.mean(np.abs((train_y,train_predict))))

print("Mape - Test:" ,np.mean(np.abs((test_y,test_predict))))