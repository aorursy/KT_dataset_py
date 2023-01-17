import pandas as pd

import numpy as np

import matplotlib as plt

import seaborn as se

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('../input/kc-housesales-data/kc_house_data.csv')
data.head()
data.dtypes
data.describe()
data.isnull().sum()

#there are no missing values
data.notnull().sum()
corr=data.corr()

se.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
data.lat.value_counts
data.long.value_counts
column_list=data.columns

df=list(set(column_list)-set(["id"])-set(["date"])-set(["price"])) #feature selection and droping id date price

x=data[df].values

y=data["price"].values

y=np.log(y) #Applying logaritm to y i.e. prices of house

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.35,random_state=0)
lr=LinearRegression(fit_intercept=True) #linear regression

model=lr.fit(xtrain,ytrain)

prediction=lr.predict(xtest)

print("Train_Accuracy")

print(lr.score(xtrain,ytrain))

print("Test_Accuracy")

print(lr.score(xtest,ytest))
regressor = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_leaf=4,min_samples_split=10,random_state=0)

model=regressor.fit(xtrain, ytrain)

y_pred = regressor.predict(xtest)

print("Train_Accuracy")

print(regressor.score(xtrain,ytrain))

print("Test_Accuracy")

print(regressor.score(xtest,ytest))
print('Mean Absolute Error:', metrics.mean_absolute_error(ytest, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(ytest, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, y_pred)))