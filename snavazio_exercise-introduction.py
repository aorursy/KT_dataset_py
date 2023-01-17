# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex1 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Obtain target and predictors

y = X_full.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = X_full[features].copy()

X_test = X_test_full[features].copy()



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_train.head()
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

print(random_grid)

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
from sklearn.metrics import mean_absolute_error

# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)

preds = rf_random.predict(X_valid)

print(mean_absolute_error(y_valid, preds))
from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))
from sklearn.ensemble import RandomForestRegressor



# Define the models

#Best at 23187...model_0 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=140)

model_1 = RandomForestRegressor(n_estimators=195, criterion='mae', random_state=140)

model_2 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=140)

model_3 = RandomForestRegressor(n_estimators=197, criterion='mae', random_state=140)

model_4 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=135)

model_5 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=136)

model_6 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=137)

model_7 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=138)

model_8 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=139)

model_9 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=140)

model_0 = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=141)



models = [model_1, model_2, model_3, model_4, model_5,model_6, model_7, model_8, model_9, model_0]
from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))
# Fill in the best model

best_model = model_3



# Check your answer

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Define a model

my_model = RandomForestRegressor(n_estimators=196, criterion='mae', random_state=140) # Your code here



# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Fit the model to the training data

my_model.fit(X, y)



# Generate test predictions

preds_test = my_model.predict(X_test)



# Save predictions in format used for competition scoring

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

#print(output)