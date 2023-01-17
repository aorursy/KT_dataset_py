from sklearn import linear_model

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
dataset=pd.read_csv("/kaggle/input/real-estate-price-prediction/Real estate.csv")

dataset.head()
dataset.isnull().sum()
x=dataset.drop("Y house price of unit area",axis=1)

y=dataset['Y house price of unit area']

y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=300)

y_test.head()
reg=linear_model.LinearRegression()

reg.fit(x_train,y_train)

reg.score(x_test,y_test)
fig,(ax1,ax2)=plt.subplots(1,2)

ax1.scatter(x_train['X2 house age'],y_train,c='r')

ax2.scatter(x_test['X2 house age'],y_test,c='y')

plt.show()
y_predict=reg.predict(x_test)

print(y_predict)
fig,(ax1,ax2)=plt.subplots(1,2)

ax1.scatter(x_test['X2 house age'],y_predict,c='r')

ax2.scatter(x_test['X2 house age'],y_test,c='y')

plt.show()


