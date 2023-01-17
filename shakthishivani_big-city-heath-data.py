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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

health_data=pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")
health_data.info()
health_data = health_data.drop(columns=["BCHC Requested Methodology","Source","Methods","Notes"])
health_data.columns
health_data["City"] = health_data["Place"].apply(lambda x:x.split('(')).str[0]
health_data.head()
health_data.isna().sum()
health_data["Value"].fillna(health_data["Value"].mean(),inplace=True)
categorical_cols = health_data.select_dtypes(exclude =np.number).columns
categorical_cols
categorical_cols = categorical_cols.drop(["Indicator","Place"])
categorical_cols
encoded_cols = pd.get_dummies(health_data[categorical_cols])
final_data = pd.concat([encoded_cols,health_data["Value"]],axis=1)
final_data.info()
y = final_data['Value']

x = final_data.drop(columns=['Value'])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.3, random_state = 42)

train_y.shape
model = LinearRegression()

model.fit(train_x,train_y)
model.coef_
model.intercept_
train_predict = model.predict(train_x)

test_predict = model.predict(test_x)
print("MSE - Train :" ,mean_squared_error(train_y,train_predict))

print("MSE - Test :" ,mean_squared_error(test_y,test_predict))

print("MAE - Train :" ,mean_absolute_error(train_y,train_predict))

print("MAE - Test :" ,mean_absolute_error(test_y,test_predict))

print("R2 - Train :" ,r2_score(train_y,train_predict))

print("R2 - Test :" ,r2_score(test_y,test_predict))

print("Mape - Train:" , np.mean(np.abs((train_y,train_predict))))

print("Mape - Test:" ,np.mean(np.abs((test_y,test_predict))))