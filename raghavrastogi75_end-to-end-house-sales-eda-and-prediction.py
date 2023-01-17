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
housing = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

print(housing.info())

#housing.head()
import seaborn as sns

housing.isna().count()
#removing id and date as they are not important for prediction

housing = housing.drop(columns = ['id','date'])

#print(housing.shape)

                        
#looking at the data types

housing.dtypes

housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))

plt.show()
housing['yr_built'].hist()

housing["year_cat"] = pd.cut(housing["yr_built"],bins=[0, 1920, 1940, 1960, 1980,2000, np.inf], labels=[1, 2, 3, 4, 5,6])

housing["year_cat"].value_counts()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index,test_index in split.split(housing, housing["year_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

# looking at the percentage wise distribution of bucket of years

strat_test_set["year_cat"].value_counts() / len(strat_test_set)

    
housing = strat_train_set.copy()

housing.plot(kind = 'scatter', x = 'long', y = 'lat', alpha = 2, figsize = (15,15),c = 'price',colorbar = True,cmap=plt.get_cmap("cool"))

corr_matrix = housing.corr()

plt.figure(figsize = (10,10))

s = corr_matrix['price'].sort_values(ascending = False)

print(s)

s.plot.bar()

import seaborn as sns

attributes = ['price','sqft_living','grade','sqft_above','sqft_living15']

housing_at = housing.loc[:,attributes]

#print(housing_at)

sns.pairplot(housing_at)

plt.show()





plt.figure(figsize = (15,15))

sns.scatterplot(x = housing['price'], y = housing['sqft_living'])
housing = strat_train_set.drop("price", axis=1)

housing_labels = strat_train_set["price"].copy()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('std_scaler', StandardScaler())])



housing_std_train = num_pipeline.fit_transform(housing)

housing_prepared = pd.DataFrame(housing_std_train, columns=housing.columns, index=housing.index)

housing_prepared

housing_labels = np.log1p(housing_labels)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

param_grid = [

{'n_estimators': [25,50], 'max_features': [8 ,10, 15]},

{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances

#print(strat_train_set.columns.tolist())

sorted(zip(feature_importances,housing.columns.tolist()),reverse = True)
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("price", axis=1)

y_test = strat_test_set["price"].copy()

X_test_prepared = num_pipeline.transform(X_test)

y_test_prep = np.log1p(y_test)

final_predictions = final_model.predict(X_test_prepared)

final_rmse = mean_squared_error(y_test_prep, final_predictions,squared = False)

print(final_rmse)
from scipy import stats

confidence = 0.95

squared_errors = (np.expm1(final_predictions) - np.expm1(y_test_prep)) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))