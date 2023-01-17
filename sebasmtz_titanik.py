import numpy as np

import pandas as pd 

import os

from sklearn.metrics import accuracy_score

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score  
titanic_file_path = '../input/train.csv'

home_data = pd.read_csv(titanic_file_path)

home_data = home_data.dropna(axis=0)



titanik_file_path = '../input/test.csv'

test_data = pd.read_csv(titanik_file_path)

test_data = test_data.dropna(axis=0)
low_cardinality_cols = [cname for cname in home_data.columns if 

                                home_data[cname].nunique() < 10 and

                                home_data[cname].dtype == "object"]

numeric_cols = [cname for cname in home_data.columns if 

                                home_data[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = home_data[my_cols]

home_data_predictions = pd.get_dummies(train_predictors)



low_cardinality_cols = [cname for cname in test_data.columns if 

                                test_data[cname].nunique() < 10 and

                                test_data[cname].dtype == "object"]

numeric_cols = [cname for cname in test_data.columns if 

                                test_data[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

test_predictors = test_data[my_cols]

test_data_predictions = pd.get_dummies(test_predictors)


feature_names = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Sex_female", "Embarked_C", "Embarked_Q", "Embarked_S"]



y = home_data_predictions.Survived

X = home_data_predictions[feature_names]



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

randomforest = RandomForestClassifier()

randomforest.fit(train_X, train_y)

y_pred = randomforest.predict(val_X)

acc_randomforest = round(accuracy_score(y_pred, val_y) * 100, 2)

print(acc_randomforest)
def get_mae(X, y):

    return -1 * cross_val_score(RandomForestRegressor(50), 

                                X, y, 

                                scoring = 'neg_mean_absolute_error').mean()



predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])



mae_one_hot_encoded = get_mae(home_data_predictions, y)



print('MAE: ' + str(int(mae_one_hot_encoded)))
ID = test_data_predictions['PassengerId']

predictions = randomforest.predict(test_data_predictions.drop('PassengerId', axis=1))



output = pd.DataFrame({ 'PassengerId' : ID, 'Survived': predictions })

output.to_csv('entregaT.csv', index=False)