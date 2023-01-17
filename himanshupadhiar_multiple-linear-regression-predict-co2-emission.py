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

print('Setup Completed')
dataset = "../input/FuelConsumptionCo2.csv"

df_data = pd.read_csv(dataset)
df_data.head()
df_data.shape
df_data.describe()
df_data.corr()
work_df = df_data[["ENGINESIZE","CYLINDERS","CO2EMISSIONS"]]
work_df.head()
work_df.corr()
msk = np.random.rand(len(work_df)) < 0.8

train_set = work_df[msk]

test_set = work_df[~msk]



print('Training Set Shape : ', train_set.shape)

print('Testing Set Shape : ', test_set.shape)



train_x = np.asanyarray(train_set[["ENGINESIZE","CYLINDERS"]])

train_y = np.asanyarray(train_set[["CO2EMISSIONS"]]).flatten()
from sklearn import linear_model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x,train_y)
r_sq = model.score(train_x,train_y)

intercept = model.intercept_

slope = model.coef_



print('r square : ', r_sq)

print('Intercept : ', intercept)

print('Slope : ', slope)
test_x = np.asanyarray(test_set[["ENGINESIZE","CYLINDERS"]])

test_y = np.asanyarray(test_set[["CO2EMISSIONS"]]).flatten()
y_pred = model.predict(test_x)

print(y_pred)
print(test_set)
r_sq_test = model.score(test_x,test_y)

print(r_sq_test)
mse = np.mean((test_y - y_pred)**2)

print('Mean Squeard Error : ', mse)