# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
housing = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
housing.shape
housing.head()
# EDA

housingc = housing.copy()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.scatter(housingc.longitude, housingc.latitude)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical distribution of houses')
plt.show()
# take a look at the density of the points
plt.figure(figsize=(10,8))
plt.scatter(housingc.longitude, housingc.latitude, alpha=0.1)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical distribution of houses')
plt.show()
correlation_matrix = housing.corr()
correlation_matrix.median_house_value.sort_values(ascending=False)
housing.isnull().sum()
# Deal with missing values of the total_bedroom  (2 ways)

# from sklearn.impute import SimpleImputer

# housing.drop(columns=['ocean_proximity'], inplace=True)  # drop column with categorical values in order for the imputer to work
# imputer = SimpleImputer()
# housing_imputed = pd.DataFrame(imputer.fit_transform(housingc))
# housing_imputed.columns = housing.columns
# put back the ocean_proximity column
# pd.concat()
# housing_imputed.isnull().sum()

housing.total_bedrooms = housing.total_bedrooms.fillna(housing['total_bedrooms'].mean())
housing.total_bedrooms.isnull().sum()
housing.ocean_proximity.nunique()
# encode ocean_proximity

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

housing['ocean_proximity'] = encoder.fit_transform(housing['ocean_proximity'])

housing.head()
from sklearn.preprocessing import StandardScaler

features = list(housing.drop('median_house_value', axis=1))
X = StandardScaler().fit(housing[features]).transform(housing[features])
y = housing['median_house_value']
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error

print('MAE: ', mean_absolute_error(y_test, model.predict(X_test)))
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=10)
model.fit(X_train, y_train)
print('MAE: ', mean_absolute_error(y_test, model.predict(X_test)))