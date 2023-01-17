# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt # visualisation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv') # trainning dataset

df_test = pd.read_csv('../input/test.csv') # testing data set
df_train.head()
df_test.head()
df_train.describe()
# getting total no. of passangers in training data

print('Total number of passengers in training data: ', len(df_train))

# getting no. of passangers who have survived

print('Total number of passengers survived: ', len(df_train[df_train['Survived'] == 1]))
# getting % of male and female survived to watch if sex played some role

print('% of male survived: ', 100*np.mean(df_train['Survived'][df_train['Sex'] == 'male']))

print('% of female survived: ', 100*np.mean(df_train['Survived'][df_train['Sex'] == 'female']))
# getting % of people who servived above anb below age 15

print('% of children who survived: ', 100*np.mean(df_train['Survived'][df_train['Age'] < 15]))

print('% of adults who survived: ', 100*np.mean(df_train['Survived'][df_train['Age'] > 15]))
# comparing survival rate of different class people

print('% of 1st class passengers who survivde: ', 100*np.mean(df_train['Survived'][df_train['Pclass'] == 1]))

print('% of 2nd class passengers who survivde: ', 100*np.mean(df_train['Survived'][df_train['Pclass'] == 2]))

print('% of 3rd class passengers who survivde: ', 100*np.mean(df_train['Survived'][df_train['Pclass'] == 3]))
# what abot SibSp and Parch

print('% of passengers with parents who survivde: ', 100*np.mean(df_train['Survived'][df_train['Parch'] >= 1]))

print('% of passengers without parents who survivde: ', 100*np.mean(df_train['Survived'][df_train['Parch'] < 1]))

print('% of passengers with siblings who survivde: ', 100*np.mean(df_train['Survived'][df_train['SibSp'] >= 1]))

print('% of passengers without siblings who survivde: ', 100*np.mean(df_train['Survived'][df_train['Parch'] < 1]))
# female = 1, male = 0

df_train['Sex'] = df_train['Sex'].apply(lambda x: 1 if x == 'female' else 0)

df_train.head()
# checking nan vlaues

df_train.isnull()

# handling nan values for age and fare

df_train['Age'] = df_train["Age"].fillna(np.mean(df_train['Age']))

df_train['Fare'] = df_train['Fare'].fillna(np.mean(df_train['Fare']))
# showing all the columns

df_train.columns
# columns which are useful for analysis

features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
# get training df

train = df_train[features]

train.head()
X = train

y = df_train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(random_state = 1)

model_dt.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, roc_curve, auc

print('Accuracy for training set: ', accuracy_score(y_train, model_dt.predict(X_train)))

print('Accuracy for testing set: ', accuracy_score(y_test, model_dt.predict(X_test)))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model_dt.predict(X_test))

print('for testing set: \n False positive rate: ', false_positive_rate, '\n True positive rate: ', true_positive_rate,

     '\n Thresholdes: ', thresholds)

roc_auc = auc(false_positive_rate, true_positive_rate)

print('ROC_AUC for testing set: ', roc_auc)
from pprint import pprint

print('Parameters currently in use: \n')

pprint(model_dt.get_params())
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]



# Create the random grid

random_grid = {'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



pprint(random_grid)
dt = DecisionTreeClassifier()



dt_randomcv = RandomizedSearchCV(estimator = dt, 

                               param_distributions = random_grid, 

                               n_iter = 100, 

                               cv = 3, verbose = 2, random_state = 1, n_jobs = -1)



dt_randomcv.fit(X_train, y_train)
dt_randomcv.best_params_
dt = DecisionTreeClassifier(random_state = 1)

# create parameter grid based on random parameter selected by random search 

param_grid = {'min_samples_split': [5, 6, 7],

              'min_samples_leaf': [3, 4, 5],

              'max_features': ['sqrt'],

              'max_depth': [85, 86, 87, 88, 90, 91, 92, 93, 94, 95]}

dt_grid_cv = GridSearchCV(estimator = dt, param_grid = param_grid,

                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data

dt_grid_cv.fit(X_train, y_train)
dt_grid_cv.best_params_
print('Accuracy for training set: ', accuracy_score(y_train, dt_grid_cv.predict(X_train)))

print('Accuracy for testing set: ', accuracy_score(y_test, dt_grid_cv.predict(X_test)))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, dt_grid_cv.predict(X_test))

print('for testing set: \n False positive rate: ', false_positive_rate, '\n True positive rate: ', true_positive_rate,

     '\n Thresholdes: ', thresholds)

roc_auc = auc(false_positive_rate, true_positive_rate)

print('ROC_AUC for testing set: ', roc_auc)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 1)

model_rf = rf.fit(X_train, y_train)
print('Accuracy for training set: ', accuracy_score(y_train, model_rf.predict(X_train)))

print('Accuracy for testing set: ', accuracy_score(y_test, model_rf.predict(X_test)))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model_rf.predict(X_test))

print('for testing set: \n False positive rate: ', false_positive_rate, '\n True positive rate: ', true_positive_rate,

     '\n Thresholdes: ', thresholds)

roc_auc = auc(false_positive_rate, true_positive_rate)

print('ROC_AUC for testing set: ', roc_auc)
print('Parameters currently in use:\n')

pprint(model_rf.get_params())
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 5000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 500, num = 10)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [int(x) for x in np.linspace(2, 16, num = 10)]

# Minimum number of samples required at each leaf node

min_samples_leaf = [int(x) for x in np.linspace(1, 15, num = 10)]

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
# using random grid search for best parameters

# create a base model to tune

rf = RandomForestClassifier()

# search across different combinations

rf_random = RandomizedSearchCV(estimator = rf,

                               param_distributions = random_grid, 

                               n_iter = 100, 

                               cv = 3, 

                               verbose = 2, 

                               random_state = 1, 

                               n_jobs = -1)

# fit the model

rf_random.fit(X_train, y_train)
# get best parameters

rf_random.best_params_
# Number of trees in random forest

n_estimators = [4200, 4500, 4800]

# Number of features to consider at every split

max_features = ['auto']

# Maximum number of levels in tree

max_depth = [50, 70, 90]

# Minimum number of samples required to split a node

min_samples_split = [8,9,10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 3]

# Method of selecting samples for training each tree

bootstrap = [True]

# Create the random grid

param_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(param_grid)
# using grid search CV to narrow down search

# create a base model to tune

rf = RandomForestClassifier(random_state = 1)

# search across different combinations

rf_gridcv = GridSearchCV(estimator = rf,

                               param_grid = param_grid, 

                               cv = 3, 

                               verbose = 2,  

                               n_jobs = -1)

# fit the model

rf_gridcv.fit(X_train, y_train)
rf_gridcv.best_params_
print('Accuracy for training set: ', accuracy_score(y_train, rf_gridcv.predict(X_train)))

print('Accuracy for testing set: ', accuracy_score(y_test, rf_gridcv.predict(X_test)))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rf_gridcv.predict(X_test))

print('for testing set: \n False positive rate: ', false_positive_rate, '\n True positive rate: ', true_positive_rate,

     '\n Thresholdes: ', thresholds)

roc_auc = auc(false_positive_rate, true_positive_rate)

print('ROC_AUC for testing set: ', roc_auc)
final_model = rf_gridcv.best_estimator_
test = df_test.copy()[features]

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'female' else 0)

test['Age'] = test["Age"].fillna(np.mean(test['Age']))

test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

final_model.fit(X,y)

predict_survival = final_model.predict(test)
submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predict_survival})



submission.to_csv('submission.csv', index=False)