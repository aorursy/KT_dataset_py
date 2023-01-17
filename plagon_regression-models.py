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
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

housing = pd.read_csv("../input/Housing.csv", index_col = 0)
housing.head(10)
housing.info()
with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'text.color' : 'white'}):

    housing.hist(bins=50, figsize = (20, 15))

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.15, random_state = 42)
housing = train_set.copy()
corr_matrix = housing.corr()

corr_matrix["price"].sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix

attributes=["price", "lotsize", "bathrms", "stories", "garagepl", "bedrooms"]

with plt.rc_context({'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor': 'white', 'text.color' : 'white'}):

    scatter_matrix(housing[attributes], figsize = (12, 8))
housing["lot_per_price"] = housing["lotsize"] / housing["price"]

housing["bedrooms_per_story"] = housing["bedrooms"] / housing["stories"]

housing["bathrms_per_story"] = housing["bathrms"] / housing["stories"]

 
corr_matrix = housing.corr()

corr_matrix["price"].sort_values(ascending=False)
housing = train_set.drop("price", axis = 1)

housing_labels = train_set["price"].copy()
from sklearn.base import BaseEstimator, TransformerMixin



class BooleanTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):

        return self

    def transform(self, X, y = None): 

        return X.replace({'yes' : 1, 'no' : 0})

        
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler



pip = Pipeline([

    ('boolean', BooleanTransformer()), 

    ('std_scaler', StandardScaler()),

])



pip2 = Pipeline([

    ('boolean', BooleanTransformer()), 

    ('minmax_scaler', MinMaxScaler()),

])
housing_prepared = pip.fit_transform(housing)

#housing_prepared
housing_prepared2 = pip2.fit_transform(housing)

#housing_prepared2
print(housing_prepared.shape)

print(housing_labels.shape)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)

#linear model on training set 

some_data = housing.iloc[:10]

some_labels = housing_labels.iloc[:10]

some_data_prepared = pip.fit_transform(some_data)

print("Predictions:\t", lin_reg.predict(some_data_prepared))

print("Labels:\t\t", list(some_labels))

#mean squared error for training data

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

tree_rmse #much better
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 

                       scoring = "neg_mean_squared_error", cv = 10)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
display_scores(rmse_scores) #for decision tree
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 

                        scoring = "neg_mean_squared_error", cv = 10)

rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 

                        scoring = "neg_mean_squared_error", cv = 10)

rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)