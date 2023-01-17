# For handling DataFrames and NumPy arrays

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Put imports for the libraries you will use here

from sklearn.ensemble import RandomForestClassifier

# End of user imports



# Render and display charts in the notebook

%matplotlib inline

# Set a white theme for seaborn charts

sns.set_style('white')
df_train = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/train.csv')

df_test = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/test.csv')
X_train = df_train.drop(['is_edible', 'id'], axis=1)

X_test = df_test.drop('id', axis=1)



y_train = df_train['is_edible']
X_train_ohe = pd.get_dummies(X_train)

X_test_ohe = pd.get_dummies(X_test)
clf = RandomForestClassifier(n_jobs=-1)

clf.fit(X_train_ohe, y_train)



y_pred = clf.predict(X_test_ohe)

y_pred
submission = pd.DataFrame(columns=['id', 'is_edible'])

submission['id'] = df_test['id']

submission['is_edible'] = y_pred

submission.to_csv('submission.csv', index=False)
# param_grid = {'parameter_name' : 'parameter_search_space',

#               'parameter_name2': 'parameter_search_space2'}

# and so on...
# If you want to use Grid search, change to GridSearchCV. Don't forget to import first!

# But be warned, it will take a long time

# param_search = RandomizedSearchCV(estimator=clf, # Your model variable goes here,

#                                   param_distributions=param_grid, # Your parameter search space

#                                   n_iter=100, # How many times do you want it to run?

#                                   cv=5,

#                                   verbose=2, # Print messages. Set to 0 to silence it

#                                   n_jobs=-1) # Use all available CPU cores