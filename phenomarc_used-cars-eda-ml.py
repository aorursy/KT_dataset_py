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
cars = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')

print('Columns:',cars.columns.tolist())

cars.info()
cars.head()
cars.drop(columns=['url', 'id', 'size', 'county', 'region_url', 'image_url', 'vin', 'description', 'state', 'lat', 'long', 'region', 'title_status'], inplace=True)
cars.info()
cars.head()
cars.describe()
cars.isnull().sum()
cars['year'].fillna(cars.year.median(), inplace=True)

cars['year']= cars.year.astype('int32')

cars['odometer'].fillna(cars.odometer.median(), inplace=True)

cars['paint_color'].fillna('Unknown', inplace=True)
cars = cars[cars['year']>=1960]

cars.year.value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")

plt.figure(figsize=(30,15))

sns.countplot(x='year', data=cars)

plt.xticks(rotation=90)

plt.xlabel('Year')

plt.ylabel('Number of offers')
plt.figure(figsize=(30,15))

sns.countplot(y='manufacturer', data=cars, order=cars['manufacturer'].value_counts().index)

plt.xlabel('Manufacturer')

plt.ylabel('Number of offers')
plt.figure(figsize=(30,15))

sns.boxplot(x='year', y='type', data=cars)
cars.paint_color.value_counts()
plt.figure(figsize=(20,10))

sns.countplot(x='paint_color', order=cars.paint_color.value_counts().index, data=cars)
cars.groupby('year').paint_color.value_counts()
reduced_cars_year_color=cars[['paint_color', 'year']]

table2=pd.pivot_table(reduced_cars_year_color, values='paint_color',index='year', columns='paint_color', aggfunc=len)
plt.figure(figsize=(20,15))

sns.heatmap(table2, annot=True, fmt='g')
cars['condition'].fillna('Unknown', inplace=True)

cars.condition.unique()
cars['manufacturer'].fillna('Unknown', inplace=True)

cars.manufacturer.unique()
plt.figure(figsize=(20,10))

sns.countplot(x='condition', order=cars.condition.value_counts().index, data=cars)

plt.xlabel('Condition')

plt.ylabel('Number of cars')
reduced_cars=cars[['condition', 'manufacturer']]

table=pd.pivot_table(reduced_cars, values='condition',index='manufacturer', columns='condition', aggfunc=len)
plt.figure(figsize=(20,10))

sns.heatmap(table, annot=True, fmt='g')
cars.info()
cars.dropna(inplace=True)

cars.info()
cars.describe()
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

categorical_columns=['year', 'drive', 'odometer', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'type', 'paint_color', 'transmission']

cars[categorical_columns] = ordinal_encoder.fit_transform(cars[categorical_columns])

cars.info()
corr_matrix = cars.corr()

corr_matrix['price']
from sklearn.model_selection import train_test_split

cars_y = cars['price']

cars_X = cars[categorical_columns]

cars_X_train, cars_X_test, cars_y_train, cars_y_test = train_test_split(cars_X, cars_y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cars_X_train = pd.DataFrame(scaler.fit_transform(cars_X_train), columns = cars_X_train.columns)

cars_X_test = pd.DataFrame(scaler.fit_transform(cars_X_test), columns = cars_X_test.columns)
cars_X_train.head()
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(cars_X_train, cars_y_train)
predictions = lin_reg.predict(cars_X_train)

from sklearn.metrics import mean_squared_error

lin_mse=mean_squared_error(cars_y_train, predictions)

lin_rmse=np.sqrt(lin_mse)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(cars_X_train, cars_y_train)
tree_predictions = tree_reg.predict(cars_X_train)

from sklearn.metrics import mean_squared_error

tree_mse=mean_squared_error(cars_y_train, tree_predictions)

tree_rmse=np.sqrt(tree_mse)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, cars_X_train, cars_y_train, scoring='neg_mean_squared_error', cv=10)

tree_rmse_scores = np.sqrt(-scores)
def display_score(scores):

    print('Scores:', scores)

    print('Mean:', scores.mean())

    print('Standard deviation:', scores.std())

display_score(tree_rmse_scores)
scores = cross_val_score(lin_reg, cars_X_train, cars_y_train, scoring='neg_mean_squared_error', cv=30)

lin_rmse_scores = np.sqrt(-scores)

display_score(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(cars_X_train, cars_y_train)
forest_predictions = forest_reg.predict(cars_X_train)

forest_mse=mean_squared_error(cars_y_train, forest_predictions)

forest_rmse=np.sqrt(forest_mse)
print("Computed RMSE's for the different models:")

print('Linear Regression Model:', lin_rmse)

print('Decision Tree Regressor Model:', tree_rmse)

print('Random Forest Regressor Model:', forest_rmse)
from sklearn.model_selection import GridSearchCV

param_grid= [

    {'max_depth': [2,4,6,8,10], 'max_features': [2,3,4]}

]

tree_reg = DecisionTreeRegressor()

grid_search = GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(cars_X_train, cars_y_train)
grid_search.cv_results_
grid_search.best_estimator_
tree_predictions = tree_reg.predict(cars_X_train)

from sklearn.metrics import mean_squared_error

tree_mse=mean_squared_error(cars_y_train, tree_predictions)

tree_rmse=np.sqrt(tree_mse)