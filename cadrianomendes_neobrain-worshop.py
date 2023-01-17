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
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
melbourne_data
melbourne_data.describe()
y = melbourne_data.Price
y
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'YearBuilt', 'Car', 'Distance', 'Propertycount',  'BuildingArea', 'Postcode']
# melbourne_features = ['Rooms', 'Postcode', 'Landsize', 'Distance']
# melbourne_features = ['Rooms', 'Distance', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X = X.fillna(0)
# X.describe()

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print('MAE', mean_absolute_error(val_y, val_predictions))
print('RMSE', mean_squared_error(val_y, val_predictions))

(pd.Series(melbourne_model.feature_importances_, index=X.columns)
   .nlargest(10)
   .plot(kind='barh'))  

'''
Linear
MAE 314184.5861893863
RMSE 342962744780.6745

Method 1
MAE 238839.28924889542
RMSE 176877324874.73196

Method 2
MAE 247851.7345285965
RMSE 160491991074.1711

Method 3
MAE 241718.3354933726
RMSE 156408696367.20856

'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 10)
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = LinearRegression()

# Fit model
melbourne_model.fit(train_X, train_y)

predicted_home_prices = melbourne_model.predict(val_X)
print('MAE', mean_absolute_error(val_y, predicted_home_prices))

predicted_home_prices = melbourne_model.predict(val_X)
print('RMSE', mean_squared_error(val_y, predicted_home_prices))
print("Making predictions for the following 5 houses:")
print(val_X.head())
print("The predictions are")
print([f'{result:.{0}f}' for result in melbourne_model.predict(val_X.head(5))])
print("Real values are")
print(val_y.head(5).values)
# Lets try "real" examples
# initialize list of lists 
data = [[1, 2, 55, -37.8072, 144.9941], 
       [2, 4, 94, -37.8072, 144.9941],
       [1, 2, 32, -37.8072, 144.9941],
       [1, 2, 23, -37.8072, 144.9941],
       [1, 2, 120, -37.8072, 144.9941],
       [1, 2, 134, -37.8072, 144.9941]]

df = pd.DataFrame(data, columns = ['Rooms', 'Distance', 'Landsize', 'Lattitude' ,'Longtitude']) 

print(melbourne_model.predict(df))
'''
melbourne_model.save()

melbourne_model.load()
'''