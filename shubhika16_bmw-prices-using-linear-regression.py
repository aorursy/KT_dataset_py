import pandas as pd

import numpy as np

import matplotlib.pyplot as plt  

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

from sklearn import metrics
#importing file

raw = pd.read_csv("../input/bmw-car-prices/bmw_carprices.csv")

print(raw.head())

#check the structure of data

print(raw.shape)

print(raw.info())

#checking stats

print(raw.describe())
#check for any missing values for each feature

print (raw.isnull().sum())
#visually finding any relationship between variables

sns.pairplot(raw)

#checking correlation coefficient

print(raw.corr())
#checking for outliers

sns.boxplot(x="variable", y="value", data=pd.melt(raw))

#definfing independent and dependent variables

x = raw[['Age(yrs)','Mileage(kms)']]

y = raw['Sell Price($)']

# standardizing the variables

stand_X = preprocessing.scale(x)

stand_y = preprocessing.scale(y)



#training the model

model = LinearRegression()

mod = model.fit(stand_X, stand_y)

print('Intercept:',model.intercept_)

# print coefficients

print(list(zip(['Age(yrs)','Mileage(kms)'], model.coef_)))

print(model.score(stand_X, stand_y)) #R-square value

y_prd=model.predict(stand_X)

#Predicted values

print(y_prd)

print('Mean Absolute Error:', metrics.mean_absolute_error(stand_y, y_prd))  

print('Mean Squared Error:', metrics.mean_squared_error(stand_y, y_prd))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(stand_y, y_prd)))