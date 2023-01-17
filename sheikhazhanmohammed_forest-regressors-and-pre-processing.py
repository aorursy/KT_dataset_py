# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("/kaggle/input/family-income-and-expenditure/Family Income and Expenditure.csv")
dataset.head()
null_data = dataset[dataset.isnull().any(axis=1)]

print(null_data.shape)
dataset = dataset.fillna(dataset.mean())
dataset = dataset.drop(['Household Head Occupation', 'Household Head Class of Worker', 'Type of Roof', 'Type of Walls', 'Toilet Facilities', 'Main Source of Water Supply'], axis= 1)
dataset = pd.get_dummies(dataset, columns=['Region','Main Source of Income','Household Head Sex', 'Household Head Marital Status', 'Household Head Highest Grade Completed', 'Household Head Job or Business Indicator', 'Type of Household', 'Type of Building/House', 'Tenure Status'])
y = dataset['Total Household Income']

dataset = dataset.drop(['Total Household Income'], axis = 1)

x = dataset
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = 0)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state=0)

regressor.fit(xTrain, yTrain)

predictedWithoutScaling = regressor.predict(xTest) 
from sklearn.metrics import r2_score, mean_squared_error

r2score = r2_score(yTest, predictedWithoutScaling)

mse = mean_squared_error(yTest, predictedWithoutScaling)

print('R2 Score using Random Forest without scaling using mean to fill NA values: ',r2score)

print('Mean Squared Error using Random Forest without scaling using mean to fill NA values: ',mse)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(xTrain)

xTrain = scaler.transform(xTrain)

xTest = scaler.transform(xTest)

regressor.fit(xTrain, yTrain)

predictedWithScaling = regressor.predict(xTest) 

r2score2 = r2_score(yTest, predictedWithScaling)

mse2 = mean_squared_error(yTest, predictedWithScaling)

print('R2 Score using Random Forest without scaling using mean to fill NA values: ',r2score2)

print('Mean Squared Error using Random Forest without scaling using mean to fill NA values: ',mse2)

from sklearn.ensemble import GradientBoostingRegressor

boostedRegressor = GradientBoostingRegressor( loss ='ls', learning_rate = 0.1, n_estimators= 200)

boostedRegressor.fit(xTrain, yTrain)

boostedPredicted = boostedRegressor.predict(xTest)

r2score3 = r2_score(yTest, boostedPredicted)

mse3 = mean_squared_error(yTest, boostedPredicted)

print('R2 Score using Gradient Boosted Forest without scaling using mean to fill NA values: ',r2score3)

print('Mean Squared Error using Gradient Boosted Forest without scaling using mean to fill NA values: ',mse3)