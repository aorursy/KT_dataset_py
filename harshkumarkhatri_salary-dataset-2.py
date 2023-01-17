from sklearn import linear_model

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
dataset=pd.read_csv("../input/salary/Salary.csv")

dataset.head()
# x=

# y=

x_train,x_test,y_train,y_test=train_test_split(dataset[['YearsExperience']],dataset[['Salary']])

x_train.head()
reg=linear_model.LinearRegression()

reg.fit(x_train,y_train)

reg.score(x_test,y_test)

# fig, (ax1,ax2,ax3)=plt.subplots(3,1)

# ax1.scatter(x_train,y_train,c='r')

plt.plot(x_test,y_test,c='b')

y_predict=reg.predict(x_test)

plt.plot(x_test,y_predict,c='y')

plt.show()