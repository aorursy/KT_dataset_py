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
housing = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import matplotlib

%matplotlib inline
housing.info()
housing['ocean_proximity'].value_counts()
housing.head()
housing.hist(bins=50,figsize = (20,20))
housing['income_cat'] = pd.cut(housing['median_income'],

                              bins = [0.,1.5,3.0,4.5,6.,np.inf],

                              labels = [1,2,3,4,5])

from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2 , random_state = 42)
for train_index , test_index in split.split(housing , housing['income_cat']):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
strat_train_set.drop("income_cat",axis = 1 ,inplace = True)

strat_test_set.drop("income_cat",axis = 1 ,inplace = True)
strat_train_set.columns
strat_test_set.columns
housing_train = strat_train_set.copy()
housing_train.plot(kind = "scatter" , x = 'longitude' , y = 'latitude')
housing_train.plot(kind = "scatter" , x = 'longitude',y = 'latitude',alpha = 0.1)
housing_train.plot(kind = "scatter" , x = 'longitude',y = 'latitude',alpha = 0.4,

                  s = housing_train['population']/100 , label = 'population' , figsize = (10,7),

                  c = 'median_house_value' , cmap = plt.get_cmap("jet"),colorbar = True)

plt.legend()
corr_mat = housing_train.corr()

corr_mat
corr_mat['median_house_value'].sort_values(ascending = False)
from pandas.plotting import scatter_matrix



attr = ['median_house_value' , 'median_income' , 'total_rooms','housing_median_age']

scatter_matrix(housing_train[attr],figsize=(15,10))
housing_train['rooms_per_household'] = housing_train['total_rooms'] / housing_train['households']

housing_train['bedrooms_per_household'] = housing_train['total_bedrooms'] / housing_train['total_rooms']

housing_train['population_per_household'] = housing_train['population'] / housing_train['households']
corr_mat = housing_train.corr()
corr_mat['median_house_value'].sort_values(ascending = False)
housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()
housing_num = housing.drop('ocean_proximity',axis = 1)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy = 'median')



X = imputer.fit_transform(housing_num)
imputer.statistics_
X
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)

housing_tr
housing_train
housing_cat = pd.DataFrame(housing_train['ocean_proximity'])

housing_cat
from sklearn.preprocessing import OneHotEncoder



onehotencoder = OneHotEncoder()

housing_cat_1hot = onehotencoder.fit_transform(housing_cat)

housing_cat_1hot
housing_cat_1hot.toarray()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

 ('imputer', SimpleImputer(strategy="median")),

 ('std_scaler', StandardScaler()),

 ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
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
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,

scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)



tree_rmse_scores
tree_rmse_scores.mean()
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,

scoring="neg_mean_squared_error", cv=10)



lin_rmse_scores = np.sqrt(-lin_scores)

lin_rmse_scores

lin_rmse_scores.mean()
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)



forest_score = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring = 'neg_mean_squared_error',cv=10)

forest_rmse_score = np.sqrt(-forest_score)

forest_rmse_score
forest_rmse_score.mean()
from sklearn.model_selection import GridSearchCV

param_grid = [

 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

 ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

 scoring='neg_mean_squared_error',

return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

     print(np.sqrt(-mean_score), params)
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse) 

final_rmse