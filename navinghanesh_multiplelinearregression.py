import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/real-estate/real_estate.csv')

data
data.describe()
x=data[['size','year']]

y=data['price']
sns.regplot(data['size'],y)
sns.regplot(data['year'],y)
regressor=LinearRegression()

regressor.fit(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y)
regressor.score(x,y)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred
regressor.score(x_train,y_train)