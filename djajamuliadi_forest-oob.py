import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pprint
%matplotlib inline
train = pd.read_csv('../input/challenge-data/train.csv', encoding='ISO-8859-1')
test = pd.read_csv('../input/challenge-data/test.csv', encoding='ISO-8859-1')
submission = pd.read_csv('../input/challenge-data/sample_submission.csv', encoding='ISO-8859-1')
train.head()
test.head()
# Check for negative prices
train.groupby(['unit_price']).size()
train[train.unit_price<0] # seemingly, these negatives are some sort of adjustments of debt

## Because there's no negatives in the test-dataset
    # test.groupby(['unit_price']).size()
# Remove negative unit price, in the training-dataset, to make life simpler. 
train = train[train.unit_price>=0]
# Start Feature Engineering
# Converting time into integer of minutes, like in TUTORIAL 3
#train['min_hour'] = train['time'].apply(lambda x: x.split(':')[0])
#train['min_minute'] = train['time'].apply(lambda x: x.split(':')[1])
#train['min_hour'] = train['min_hour'].apply(lambda x : int(x) * 60 )
#train['min_minute'] = train['min_minute'].apply(lambda x: int(x))
#train['tot_min'] = train['min_hour'] + train['min_minute']
# So, these are the only columns I'll use as predictors
pred_cols = ['unit_price','customer_id','country', 'stock_id']
train_candidates = train[pred_cols]


# One-hot-encode Country
onehotencoded_candidates = pd.get_dummies(train_candidates)
onehotencoded_candidates.shape
train.info()
# Check for uniqueness of each relevant variable
train.apply(pd.Series.nunique).sort_values()
# Looks like country has the least number of uniqueness from original data. This might qualify for 1-hot-encoding(?), as described in the blogs above.
# Here is my training dataset
train_X = onehotencoded_candidates
train_y = train['quantity']
# Splitting this training dataset further: to make train_train and train_test. This way I'd be able check for the "goodness of learning"
from sklearn.model_selection import train_test_split
train_train_X, train_test_X, train_train_y, train_test_y = train_test_split(train_X, train_y,random_state=0)

### Using RMSLE as validation

def rmsle(y_true,y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))
## Try Random Forests "out of the box"
from sklearn.ensemble import RandomForestRegressor

Forest_model_oob = RandomForestRegressor(n_jobs =-1, random_state=0)
Forest_model_oob.fit(train_train_X,train_train_y)
train_test_predictions = Forest_model_oob.predict(train_test_X)
print('Using Forest_model_oob, the rmsle is : {} \n'.format(rmsle(train_test_y,
                                                               train_test_predictions)))
pprint.pprint(Forest_model_oob.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of n_estimators
n_est = [60]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Number of max_leaf_nodes
max_leaf_nodes = [5,1000,5000, 7500, 10000]
# how deep?
max_depth = [10, 25, 50, 75, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 50, 75]
# Create the random grid
random_grid = {'max_features': max_features,
               'n_estimators': n_est,
               'max_leaf_nodes': max_leaf_nodes,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
               }
pprint.pprint(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 25 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3,
                               verbose=2, random_state=0, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_train_X, train_train_y)
rf_random.best_params_
def rmsle(y_true,y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))
def evaluate(model, train_test_X, train_test_y):
    predictions = model.predict(train_test_X)
    errors = abs(predictions - train_test_y)
    print('Model Performance:')
    print('Average deviation: {:0.1f} from actual quantity.'.format(np.mean(errors))) 
    print('rmsle= {:0.6f}'.format(rmsle(train_test_y,predictions)))
    return rmsle(train_test_y,predictions)

base_model = RandomForestRegressor(random_state = 0, n_jobs=-1)
base_model.fit(train_train_X, train_train_y)
base_rmsle = evaluate(base_model, train_test_X, train_test_y)
best_random = rf_random.best_estimator_
random_rmsle = evaluate(best_random, train_test_X, train_test_y)

print('With Random_search, the RMSLE is improved by {:0.1f}%.'.format( 100 * (random_rmsle - base_rmsle) / base_rmsle))
test_candidates = test[pred_cols] # including the needed columns in test 
test_X = pd.get_dummies(test_candidates) # onehotcoding country
test_prediction = base_model.predict(test_X)
my_submission = pd.DataFrame({'id':test.id,
                             'quantity':test_prediction})
my_submission.to_csv('submission.csv', index=False)
#from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
#param_grid = {
#    'max_depth': [75, 100, 125],
#    'max_features': [2, 3],
#    'min_samples_leaf': [1, 2, 3],
#    'max_leaf_nodes': [6000, 7500, 8000],
#    'n_estimators': [50, 75, 90]
#}
# Create a based model
#rf = RandomForestRegressor()
# Instantiate the grid search model, 2-fold CV
#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
#grid_search.fit(train_train_X, train_train_y)
#grid_search.best_params_
#{'max_depth': 100,
# 'max_features': 'auto',
# 'min_samples_leaf': 2,
# 'min_samples_split': 7500,
# 'n_estimators': 60}

#best_grid = grid_search.best_estimator_
#grid_rmsle = evaluate(best_grid, train_test_X, train_test_y)


#print('With grid_search after random_search, the RMSLE is improved by {:0.1f}%.'.format( 100 * (random_rmsle - base_rmsle) / base_rmsle))
