### Importing required packages
import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=df_test['Unnamed: 0'] #copying test index for later
import numpy as np

df_train.head()

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 90, stop = 290, num = 7)]

# Number of features to consider at every split

max_features = ['sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 13, num = 5)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [ 6, 10,14,18]

# Minimum number of samples required at each leaf node

min_samples_leaf = [ 5, 9, 13,17]

# Method of selecting samples for training each tree

bootstrap = [True]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)

train_X = df_train.loc[:, ['V6','V12','V14','V15']]

train_y = df_train.loc[:, 'Class']


w=18

rf = RandomForestClassifier(n_estimators=50, random_state=123,class_weight={0: 1, 1: w})
w=18

# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier(class_weight={0: 1, 1: w})

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 108, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(train_X, train_y)



rf_random.best_params_

def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy
df_test = df_test.loc[:, ['V6','V12','V14','V15']]

pred = rf.predict_proba(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)