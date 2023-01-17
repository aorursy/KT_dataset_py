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
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from pprint import pprint
test = pd.read_csv("../input/iowa-house-prices/test.csv")

train = pd.read_csv("../input/iowa-house-prices/train.csv")
train.head()
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = train[features]

print(X.head())

y = train['SalePrice']

print(y.head())
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf = RandomForestRegressor(random_state=1)



print('Parameters currently in use:\n')

pprint(rf.get_params())
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

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1, n_jobs = -1)

# Fit the random search model

rf_random.fit(train_X, train_y)
pprint(rf_random.best_params_)
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    mae = mean_absolute_error(predictions, test_labels)

    print('Model Performance')

    print('Average Error: {:0.4f}'.format(np.mean(mae)))    

    return mae
base_model = RandomForestRegressor(n_estimators=10, random_state=1)

base_model.fit(train_X, train_y)

base_mae = evaluate(base_model, val_X, val_y)
best_random = rf_random.best_estimator_

best_random_mae = evaluate(best_random, val_X, val_y)
# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(train_X, train_y)

grid_search.best_params_
best_grid = grid_search.best_estimator_

grid_search_mae = evaluate(best_grid, val_X, val_y)
test.head()
test_X = test[features]

test_predictions = best_random.predict(test_X)

output = pd.DataFrame({'ID': test.Id, 'SalePrice': test_predictions})

output.to_csv('submission.csv', index=False)