# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cars_filepath = "../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv"

cars = pd.read_csv(cars_filepath)
cars.columns
cars.shape
cars.info()
cars.head(10)
cars.describe()
cars.name.describe()
plt.figure(figsize = (10,10))

sns.barplot(x = 'owner', y = 'selling_price', data = cars)
plt.figure(figsize = (10,10))

sns.barplot(x = 'fuel', y = 'selling_price', data = cars)
plt.figure(figsize = (10,5))

plt.ylim(0,700000)

plt.xlim(0,800000)

sns.regplot(x = 'km_driven', y = 'selling_price', data = cars, color = 'green')
cars_copy = cars.copy()
lst = (cars_copy.dtypes == 'object')

object_col = list(lst[lst].index)

print(object_col)
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

for col in object_col:

    cars_copy[col] = encoder.fit_transform(cars_copy[col])
attributes = ['name','year', 'km_driven', 'fuel', 'seller_type',

       'transmission', 'owner']

X = cars_copy[attributes]

y = cars_copy.selling_price
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.6, random_state = 0)
from sklearn.linear_model import LinearRegression



regressor = LinearRegression()

regressor.fit(X_train,y_train)



print(regressor.intercept_)

print(regressor.coef_)
y_pred = regressor.predict(X_valid)

print(y_pred)

print(y_valid)
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_valid, y_pred))
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_leaf_nodes = 700, random_state = 0)

tree.fit(X_train,y_train)

tree_y_pred = tree.predict(X_valid)

tree_y_pred

print(mean_absolute_error(y_valid,tree_y_pred))
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(random_state=1)

forest.fit(X_train, y_train)

forest_y = forest.predict(X_valid)

print(mean_absolute_error(y_valid, forest_y))