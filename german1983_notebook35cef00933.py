# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Read the train data
train = pd.read_csv('../input/train.csv')
# Read the test data
test = pd.read_csv('../input/test.csv')

print('Train Columns:', train.columns)
print('Test Columns:', test.columns)

# pull data into target (y) and predictors (X)
train_y = train.Survived
predictor_cols = train.drop(['Survived'], axis=1).drop(['PassengerId'], axis=1).select_dtypes(exclude=['object']).columns
print(predictor_cols)
# Create training predictors data
train_X = train[predictor_cols]
test_X = test[predictor_cols]
print(train_X.dtypes)
print(test_X.dtypes)

# Import imputer to impute missing values
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(test_X)


# ImportRandom Forest
from sklearn.ensemble import RandomForestRegressor
my_model = RandomForestRegressor(random_state=0)
my_model.fit(imputed_X_train, train_y)
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(imputed_X_train, train_y, test_size=0.25)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
print(my_model.best_ntree_limit)
print(my_model.best_iteration)
print(my_model.best_score)

my_best_model = XGBRegressor(n_estimators=my_model.best_ntree_limit, learning_rate=0.05)
my_best_model.fit(train_X, train_y)
# Use the model to make predictions
predicted_survival = my_best_model.predict(imputed_X_test)
print(predicted_survival)
predicted_survival_binary = [int(round(x)) for x in predicted_survival]
print(predicted_survival_binary)
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted_survival_binary})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
print("Success")