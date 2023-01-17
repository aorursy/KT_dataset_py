import os

import numpy as np

import pandas as pd
data = pd.read_csv("/kaggle/input/life-expectancy-who/Life Expectancy Data.csv")
data.head()
data = data.fillna(data.mean())
columns = data.columns
unique_years=(data["Year"].unique())

unique_status = data["Status"].unique()

unique_status
data["Year"]=data.Year.astype("category").cat.codes

data["Status"]=data.Status.astype("category").cat.codes

data["Country"] = data.Country.astype("category").cat.codes
data.head()
import matplotlib.pyplot as plt

import seaborn as sns
data["Target"]=data[columns[3]]
data=data.drop(columns=columns[3],axis=1)

data.head()
from sklearn.model_selection import train_test_split
data_x = data.drop(columns="Target",axis=1)

data_y = data["Target"]
x_train , x_test , y_train , y_test = train_test_split(data_x,data_y, test_size = 0.2)

print(x_train.shape , x_test.shape , y_train.shape , y_test.shape)
x_train = x_train.to_numpy()

x_test  = x_test.to_numpy()

y_train = y_train.to_numpy()

y_test  = y_test.to_numpy()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
y_train.resize((y_train.shape[0],1))
model.fit(x_train,y_train)
y_train_predict = model.predict(x_train)

y_train_predict
y_train
train_result = pd.DataFrame(data=np.column_stack((y_train,y_train_predict)),columns=['Y','P'])
train_result["error % "] = (abs(train_result["Y"]-train_result["P"])/train_result["Y"])*100

train_result
print(train_result["error % "].mean())
y_test_predict = model.predict(x_test)
y_test.resize((y_test.shape[0],1))
test_result = pd.DataFrame(data=np.column_stack((y_test,y_test_predict)),columns=['Y','P'])

test_result["error % "] = (abs(test_result["Y"]-test_result["P"])/test_result["Y"])*100

test_result
print(test_result["error % "].mean())