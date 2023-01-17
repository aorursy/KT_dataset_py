#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# This is a minimal script to perform a classification on the kaggle 

# 'Titanic' data set using XGBoost Python API 

# Carl McBride Ellis (12.IV.2020)

#===========================================================================

#===========================================================================

# load up the libraries

#===========================================================================

import pandas  as pd

import xgboost as xgb

from   xgboost import XGBClassifier



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')

test_data  = pd.read_csv('../input/titanic/test.csv')



#===========================================================================

# select some features of interest ("ay, there's the rub", Shakespeare)

#===========================================================================

features = ["Pclass", "Sex", "SibSp", "Parch"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies:

# "Convert categorical variable into dummy/indicator variables."

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]

final_X_test  = pd.get_dummies(test_data[features])



#===========================================================================

# XGBoost classification: 

# Parameters: 

# n_estimators  "Number of gradient boosted trees. Equivalent to number of 

#                boosting rounds."

# learning_rate "Boosting learning rate (xgb’s “eta”)"

# max_depth     "Maximum depth of a tree. Increasing this value will make 

#                the model more complex and more likely to overfit." 

#===========================================================================

classifier = XGBClassifier(n_estimators=750,learning_rate=0.02,max_depth=3)

classifier.fit(X_train, y_train)



#===========================================================================

# use the model to predict 'Survived' for the test data

#===========================================================================

predictions = classifier.predict(final_X_test)



#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)