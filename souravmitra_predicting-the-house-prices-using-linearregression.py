import pandas as pd

import numpy as np
import os

print(os.listdir("../input/california-housing-prices"))
housing_data = pd.read_csv('../input/california-housing-prices/housing.csv')
housing_data.info()
housing_data.isnull().sum()[housing_data.isnull().sum() > 0]
housing_data.fillna(value = housing_data['total_bedrooms'].mean(), axis = 1, inplace = True) 
ocean_proximity = pd.get_dummies(housing_data['ocean_proximity'])
housing_data = housing_data.join(ocean_proximity)
housing_data.drop(['ocean_proximity'], axis = 1, inplace = True)
x = housing_data.drop('median_house_value', axis = 1)
y = housing_data['median_house_value']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing_data_scaled = scaler.fit_transform(x)
housing_data_final = pd.DataFrame(data = housing_data_scaled, columns = [['longitude', 'latitude', 'housing_median_age', 'total_rooms',

       'total_bedrooms', 'population', 'households', 'median_income',

       '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']] )
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
x1 = housing_data_final

y1 = housing_data['median_house_value']
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state = 101)
param = {'n_jobs' : [ 0.0001, 0.001, 0.001, 0.01]}
GridSearch = GridSearchCV(estimator = LinearRegression(), param_grid = param, verbose = 5)
GridSearch.fit(x_train, y_train)
GridSearch.best_estimator_
LinearModel = LinearRegression(n_jobs = 0.0001)
LinearModel.fit(x_train, y_train)
predictions = LinearModel.predict(x_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(y_test, predictions))
print(np.sqrt(mean_squared_error(y_test,predictions)))