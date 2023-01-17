# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# save filepath to variable for easier access

melbourne_file_path = '../input/Melbourne_housing_FULL.csv'

# read the data and store data in DataFrame titled melbourne_data

melbourne_data = pd.read_csv(melbourne_file_path) 



melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 

                        'YearBuilt', 'Lattitude', 'Longtitude']

#Droping all Rows with null values

melbourne_data = melbourne_data.dropna()



melbourne_data.describe()

# Any results you write to the current directory are saved as output.
 #store the series of prices separately as melbourne_price_data.

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

Y = melbourne_data.Price

X = melbourne_data[melbourne_predictors]



X = pd.get_dummies(X)

# split data into training and validation data, for both predictors and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, Y,random_state = 0)

# Define model

melbourne_model = make_pipeline(Imputer(),DecisionTreeRegressor())

# Fit model

melbourne_model.fit(train_X, train_y)
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, X, Y, scoring='neg_mean_absolute_error')

print('Mean Absolute Error %2f' %(-1 * scores.mean()))
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)

mean_absolute_error(Y, predicted_home_prices)

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):

    model =make_pipeline(Imputer(), DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0))

    model.fit(predictors_train, targ_train)

    preds_val = model.predict(predictors_val)

    mae = mean_absolute_error(targ_val, preds_val)

    return(mae)
# compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = make_pipeline(Imputer(),RandomForestRegressor())

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, melb_preds))
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000,learning_rate=0.05)

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X, train_y, early_stopping_rounds=100,

             eval_set=[(val_X, val_y)], verbose=False)
# make predictions

predictions = my_model.predict(val_X)

print(val_X.head())

print(predictions)

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, val_y)))
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.preprocessing import Imputer



my_model = GradientBoostingRegressor()

my_model.fit(train_X, train_y)

my_plots = plot_partial_dependence(my_model, 

                                   features=[0,6], 

                                   X=train_X, 

                                   feature_names=melbourne_predictors, 

                                   grid_resolution=10)

    