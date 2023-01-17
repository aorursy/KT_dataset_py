# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
house_sales = pd.read_csv("../input/kc_house_data.csv")

house_sales
house_sales.columns
y = house_sales.price

predictors = ['bedrooms','bathrooms','view','yr_built','sqft_basement']

x = house_sales[predictors]

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(x,y)

from sklearn.metrics import mean_absolute_error

predict_house_price = model.predict(x)

mae = mean_absolute_error(y,predict_house_price)

print(mae)
from sklearn.model_selection import train_test_split

train_X, val_X, train_Y, val_Y = train_test_split(x,y,random_state=0)

model = DecisionTreeRegressor()

model.fit(train_X,train_Y)

val_prediction = model.predict(val_X)

mer = mean_absolute_error(val_prediction,val_Y)

print(mer)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor()

model.fit(train_X,train_Y)

val_prediction = model.predict(val_X)

mer = mean_absolute_error(val_prediction,val_Y)

print(mer)