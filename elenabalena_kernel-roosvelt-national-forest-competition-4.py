# This kernel is used to make predictions on what types of forest is in an area based on various geographic features. 

# The data comes from the Roosvelt National Forest in Colorado.



# Input data files are available in the "../input/" directory.

# Read the test and training data



import pandas as pd

X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')

X_full = pd.read_csv('/kaggle/input/learn-together/train.csv',index_col='Id')
# Separate the data into a training and a validation set



y = X_full.Cover_Type

X_full.drop(['Cover_Type'], axis=1, inplace=True)



from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y,

                                                     train_size=0.8, test_size=0.2,

                                                     random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



rf_mod = RandomForestClassifier()



# Specify the parameters of the grid

param = {'n_estimators' :[250,500,1000,1500],

         'max_depth' : [10,50,100,None],

         'min_samples_split' : [2, 5, 10],

         'min_samples_leaf' : [1, 2, 4],

         'bootstrap': [True, False]}



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf_mod, param_grid = param, 

                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the model



grid_search.fit(X_train, y_train)
# Make predictions



best_grid = grid_search.best_estimator_

y_pred = best_grid.predict(X_valid)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_valid, y_pred)

print(accuracy)
# Make predictions on testb data

predictions = best_grid.predict(X_test)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'Cover_Type': predictions})

output.to_csv('submission4.csv', index=False)
