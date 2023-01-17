# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
ins_df = pd.read_csv("../input/insurance.csv")
ins_df.head()
ins_df.isna().sum()
#no null values are there

ins_df.tail()
ins_df.sex.unique()
num_col = ins_df.select_dtypes(include=np.number).columns

num_col
cat_col = ins_df.select_dtypes(exclude=np.number).columns

cat_col
#one hot encoding 

encoded_cat_col = pd.get_dummies(ins_df[cat_col])

encoded_cat_col
ins_df_ready_model = pd.concat([ins_df[num_col],encoded_cat_col], axis=1)

ins_df_ready_model

ins_df_ready_model.corr()
y_axis = ins_df_ready_model['expenses']

x_axis = ins_df_ready_model.drop(columns='expenses')
train_x,test_x,train_y,test_y = train_test_split(x_axis,y_axis, test_size=0.3)
from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(train_x,train_y)
train_predict = model.predict(train_x)



test_predict = model.predict(test_x)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



MAE_train = mean_absolute_error(train_y,train_predict)

MAE_test = mean_absolute_error(test_y,test_predict)



MSE_train = mean_squared_error(train_y,train_predict)

MSE_test = mean_squared_error(test_y,test_predict)



RMSE_train = np.sqrt(MSE_train)

RMSE_test = np.sqrt(MSE_test)



Mape_train = np.mean(np.abs((train_y,train_predict)))

Mape_test = np.mean(np.abs((test_y,test_predict)))



print("MAE of Trained data : ",MAE_train)

print("MAE of Test data    : ", MAE_test)



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("MSE of Trained Data", MSE_train)

print("MSE of Test Data", MSE_test)



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



print("RMSE of Trained Data", RMSE_train)

print("RMSE of Test Data", RMSE_test)



print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print("Mape of train :, ",Mape_train)

print("Mape of test :, ",Mape_test)



import matplotlib.pyplot as plot

%matplotlib inline



plot.figure(figsize=(10,7))

plot.title("Actual vs. predicted",fontsize=25)

plot.xlabel("Actual",fontsize=18)

plot.ylabel("Predicted", fontsize=18)

plot.scatter(x=test_y,y=test_predict)