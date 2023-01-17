#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# This is a simple script to perform a classification on the kaggle 

# 'Titanic' data set using a grid search, in conjunction with a 

# random forest classifier

# Carl McBride Ellis (1.V.2020)

#===========================================================================

#===========================================================================

# load up the libraries

#===========================================================================

import pandas as pd

import numpy  as np



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')

test_data  = pd.read_csv('../input/titanic/test.csv')



#===========================================================================

# select some features of interest ("ay, there's the rub", Shakespeare)

#===========================================================================

features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies:

# "Convert categorical variable into dummy/indicator variables."

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]

final_X_test  = pd.get_dummies(test_data[features])



#===========================================================================

# hyperparameter grid search using scikit-learn GridSearchCV

# we use the default 5-fold cross validation

#===========================================================================

from sklearn.model_selection import GridSearchCV

# we use the random forest classifier

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion='gini', max_features='auto')

gs = GridSearchCV(cv=5, error_score=np.nan, estimator=classifier,

# dictionaries containing values to try for the parameters

param_grid={'min_samples_leaf':  [20, 25, 30],

            'min_samples_split': [2, 5, 7, 10],

            'max_depth':         [3, 4, 5, 6],

            'n_estimators':      [3, 5, 10]})

gs.fit(X_train, y_train)



# grid search has finished, now echo the results to the screen

print("The best score is ",gs.best_score_)

print("The best parameters are ",gs.best_params_)

the_best_parameters = gs.best_params_



#===========================================================================

# now perform the final fit, using the best values from the grid search

#===========================================================================

classifier = RandomForestClassifier(criterion='gini', max_features='auto',

             min_samples_leaf  = the_best_parameters["min_samples_leaf"],

             min_samples_split = the_best_parameters["min_samples_split"],

             max_depth         = the_best_parameters["max_depth"],

             n_estimators      = the_best_parameters["n_estimators"])

classifier.fit(X_train, y_train)



#===========================================================================

# use the model to predict 'Survived' for the test data

#===========================================================================

predictions = classifier.predict(final_X_test)



#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 

                       'Survived': predictions})

output.to_csv('submission.csv', index=False)