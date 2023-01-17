

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

home_data_path = '../input/home-data-for-ml-course/train.csv'

test_x ='../input/home-data-for-ml-course/test.csv'



home_data = pd.read_csv(home_data_path)

test = pd.read_csv(test_x)

features_taken = []

for column in home_data:

    if home_data[column].dtypes == 'int64':

        features_taken.append(column)

features_taken.remove('SalePrice')

print(features_taken)
X = home_data[features_taken]

y = home_data.SalePrice



test_X = home_data[features_taken]



home_model = RandomForestRegressor(random_state = 1)

home_model.fit(X, y)
sale_prediction = home_model.predict(test_X)

print(mean_absolute_error(y, sale_prediction))