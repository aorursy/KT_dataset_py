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
data=  pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")

data.head()
data.describe()
# Libraries

import plotly.express as px
data.info()
# Housing Median Age

fig = px.histogram(data, x="housing_median_age", nbins = 30)

fig.show()
# Total Rooms

fig = px.histogram(data, x="total_rooms")

fig.show()
# Total Bed Rooms

fig = px.histogram(data, x="total_bedrooms")

fig.show()
# Population

fig = px.histogram(data, x="population")

fig.show()
# Households

fig = px.histogram(data, x="households")

fig.show()
# median_income

fig = px.histogram(data, x="median_income")

fig.show()
# median_house_value

fig = px.histogram(data, x="median_house_value")

fig.show()
# Ocean Proximity

df = data.groupby('ocean_proximity')['ocean_proximity'].count().reset_index(name = 'count')

fig = px.bar(df, x='ocean_proximity', y='count')

fig.show()
fig = px.scatter(data, x="median_house_value", y="ocean_proximity", color = 'housing_median_age')

fig.show()
fig = px.scatter(data, x="median_house_value", y="housing_median_age", color = 'ocean_proximity')

fig.show()
fig = px.scatter(data, x="longitude", y="latitude", color = 'ocean_proximity')

fig.show()
fig = px.scatter(data, x="longitude", y="latitude", color = 'median_house_value')

fig.show()
df_corr = data.corr()

fig = px.imshow(df_corr)

fig.show()
from pandas.plotting import scatter_matrix

attr = ['median_house_value','median_income','total_rooms','housing_median_age']

scatter_matrix(data[attr], figsize = (15,10))
data["income_cat"] = pd.cut(data["median_income"],

                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],

                               labels=[1, 2, 3, 4, 5])
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data,data['income_cat']):

    strat_train_set = data.loc[train_index]

    strat_test_set = data.loc[test_index]

print(strat_train_set.shape)

print(strat_test_set.shape)

print(data.shape)
# Creating new features from existing

data['rooms_per_household'] = data['total_rooms'] / data['households']

data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']

data['population_per_household'] = data['population'] / data['households']
corr = data.corr()

corr['median_house_value'].sort_values(ascending = False)
fig = px.imshow(corr)

fig.show()
from sklearn.impute import SimpleImputer
imputer  = SimpleImputer(strategy = 'median')

housing_num = data.drop("ocean_proximity", axis = 1)

median = imputer.fit(housing_num)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
from sklearn.preprocessing import OneHotEncoder

housing_cat = data[['ocean_proximity']]

cat_encoder = OneHotEncoder()

housing_onehot = cat_encoder.fit_transform(housing_cat)
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set

housing_labels = strat_train_set["median_house_value"].copy()
features = strat_train_set.columns.to_list()
# Custom Transformers to create some extra features.



from sklearn.base import BaseEstimator, TransformerMixin



# column index

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

                         bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



housing_num = housing.drop("ocean_proximity", axis=1)

num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

    ])



housing_num_tr = num_pipeline.fit_transform(housing_num)
# add the categorical columns with transformer

from sklearn.compose import ColumnTransformer



num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



housing_prepared = full_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)



print("Predictions:", lin_reg.predict(some_data_prepared))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor().fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

print(tree_rmse)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)

tree_rmse_scores = np.sqrt(-scores)

print("Scores : ",str(tree_rmse_scores))

print("Mean : ",str(tree_rmse_scores.mean()))

print("Standard Deviation : ",str(tree_rmse_scores.std()))
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)

lin_rmse_scores = np.sqrt(-lin_scores)

print("Scores : ",str(lin_rmse_scores))

print("Mean : ",str(lin_rmse_scores.mean()))

print("Standard Deviation : ",str(lin_rmse_scores.std()))
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(housing_prepared, housing_labels)



forest_scores = cross_val_score(forest, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)

forest_rmse_scores = np.sqrt(-forest_scores)

print("Scores : ",str(forest_rmse_scores))

print("Mean : ",str(forest_rmse_scores.mean()))

print("Standard Deviation : ",str(forest_rmse_scores.std()))
from sklearn.model_selection import GridSearchCV

param_grid = [

    {'n_estimators':[10,30, 50, 70, 100],'max_features':[6,8,10,15]},

    {'bootstrap':[False], 'n_estimators':[3,10],'max_features':[2, 3, 4]}]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, 

                           param_grid, 

                           cv = 5, 

                           scoring = 'neg_mean_squared_error', 

                           return_train_score = True)

grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
grid_search.best_estimator_
st_time = time.time()

forest = RandomForestRegressor(max_features = 8)

forest.fit(housing_prepared, housing_labels)



forest_scores = cross_val_score(forest, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)

forest_rmse_scores = np.sqrt(-forest_scores)

print("Scores : ",str(forest_rmse_scores))

print("Mean : ",str(forest_rmse_scores.mean()))

print("Standard Deviation : ",str(forest_rmse_scores.std()))

end_time = time.time()

print("Total Time : ",str(round(end_time - st_time,2)))
import time

st_time = time.time()

forest = RandomForestRegressor(max_features = 15)

forest.fit(housing_prepared, housing_labels)



forest_scores = cross_val_score(forest, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)

forest_rmse_scores = np.sqrt(-forest_scores)

print("Scores : ",str(forest_rmse_scores))

print("Mean : ",str(forest_rmse_scores.mean()))

print("Standard Deviation : ",str(forest_rmse_scores.std()))

end_time = time.time()

print("Total Time : ",str(round(end_time - st_time,2)))