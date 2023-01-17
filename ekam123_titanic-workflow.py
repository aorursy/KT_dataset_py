# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


train= pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_two = train.drop(['PassengerId','Survived', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
y = train['Survived']
test_data = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
train_two_predictors = pd.get_dummies(train_two)
test_data_predictors = pd.get_dummies(test_data)
test_data_predictors = test_data_predictors.as_matrix()
train_X, test_X, train_y, test_y = train_test_split(train_two_predictors.as_matrix(), y.as_matrix(), test_size=0.25)
my_model = make_pipeline(Imputer(),XGBClassifier(n_estimators=1000, learning_rate=0.05))
my_model.fit(train_X, train_y)#, early_stopping_rounds=5,verbose=False)
# make predictions
predictions = my_model.predict(test_X)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_y, predictions))
predictionstest = my_model.predict(test_data_predictors)
my_submission5 = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictionstest})
my_submission5.to_csv('submission5.csv', index=False)
