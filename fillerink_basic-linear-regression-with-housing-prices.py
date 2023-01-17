import os

import pandas as pd

import numpy as np

import matplotlib as plt



from sklearn.model_selection import train_test_split


housing = pd.read_csv('../input/housing.csv')



train,test = train_test_split(housing,test_size=0.33,random_state=42)





train.head()
train = train.drop('ocean_proximity',axis=1)

test = test.drop('ocean_proximity',axis=1)

train = train.drop('total_bedrooms',axis=1)

test = test.drop('total_bedrooms',axis=1)



train.head()
x_train = train.drop('median_house_value',axis=1)
y_train = train.median_house_value
x_train.head()
y_train.head()
# first import the function from scikit-learn

from sklearn.linear_model import LinearRegression
# create a new object of Linear Regression class

model = LinearRegression()
# fitting the model = finding the perfect line with minimum error

model.fit(x_train,y_train)
model.score(x_train,y_train)
x_test = test.drop('median_house_value',axis=1)
model.predict(x_test)
test.median_house_value
from sklearn.ensemble import RandomForestRegressor
my_new_model = RandomForestRegressor()
my_new_model.fit(x_train,y_train)
my_new_model.score(x_train,y_train)
output = my_new_model.predict(x_test)
my_new_model.score(x_test,test.median_house_value)
output_csv = pd.DataFrame({'Label':output})



output_csv.to_csv('output.csv',index=False)