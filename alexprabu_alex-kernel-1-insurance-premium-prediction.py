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
df = pd.read_csv("../input/insurance.csv")
df.head()
df.shape
df.isna().sum()
num_col = df.select_dtypes(include=np.number).columns

num_col
cat_col = df.select_dtypes(exclude=np.number).columns

cat_col
encoded_cat_col = pd.get_dummies(df[cat_col])
encoded_cat_col
df_preprocesssed = pd.concat([df[num_col], encoded_cat_col], axis = 1)
df_preprocesssed
x = df_preprocesssed.drop(columns=['expenses'])

y = df_preprocesssed[['expenses']]
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state = 1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)
model.intercept_
model.coef_
predict_train = model.predict(train_x)

predict_test = model.predict(test_x)
from sklearn.metrics import mean_absolute_error, mean_squared_error
train_MSE = mean_squared_error(train_y, predict_train)

test_MSE = mean_squared_error(test_y, predict_test)
train_MAE = mean_absolute_error(train_y, predict_train)

test_MAE= mean_absolute_error(test_y, predict_test)
train_RMSE = np.sqrt(train_MSE)

test_RMSE = np.sqrt(test_MSE)
train_MAPE = np.mean(np.abs(train_y, predict_train))

test_MAPE = np.mean(np.abs(test_y, predict_test))
print("train_MAE: ",train_MAE)

print("test_MAE: ",test_MAE)
print("train_MSE: ",train_MSE)

print("test_MSE: ",test_MSE)
print("train_RMSE: ", train_RMSE)

print("test_RMSE: ", test_RMSE)
print("train_MAPE: ", train_MAPE)

print("test_MAPE: ", test_MAPE)