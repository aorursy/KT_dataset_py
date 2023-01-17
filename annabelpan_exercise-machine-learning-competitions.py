import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

print("the full dataset dimension is :{}".format(home_data.shape))

home_data.head(5)

home_data.describe()
df_num = home_data.select_dtypes(exclude=['object'])

print(df_num.shape)

display(df_num.head(5))


df_num.isnull().sum()

print(df_num.isnull().sum()/len(df_num)*100)
df_num.fillna(df_num.mean(), inplace=True)
import seaborn as sb

sb.heatmap(df_num.isnull(), cbar=False)
X=df_num.drop('SalePrice', axis=1)

y = df_num['SalePrice']




# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=42)



from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf_model_on_full_data.get_params())





import numpy as np

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

#rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_model_on_full_data = RandomizedSearchCV(estimator = rf_model_on_full_data, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_model_on_full_data.fit(X, y)
rf_model_on_full_data.best_params_
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

print(test_data.shape)



# only consider numerical variables

df_test_num = test_data.select_dtypes(exclude=['object'])

print(df_test_num.shape)

display(df_test_num.head(5))







df_test_num.isnull().sum()

print(df_test_num.isnull().sum()/len(df_test_num)*100)
sb.heatmap(df_test_num.isnull(), cbar=False)
df_test_num.fillna(df_test_num.mean(), inplace=True)
sb.heatmap(df_test_num.isnull(), cbar=False)

df_test_num.isnull().sum()
test_X=df_test_num
# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)
# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
step_1.check()

step_1.solution()