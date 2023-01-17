import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_y = train.Survived
low_cardinality_cols = [cname for cname in train.columns if 
                                train[cname].nunique() < 10 and
                                train[cname].dtype == "object"]
my_columns = low_cardinality_cols + [cname for cname in train.columns if train[cname].dtype in ['int64', 'float64']]
my_columns.remove('Survived')
train_predictors = train[my_columns]
test_predictors = test[my_columns]

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
train_X, test_X = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='left', axis=1)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, eval_set=[(train_X[0:100], train_y[0:100])], early_stopping_rounds=5, verbose=False)

predicted_survival = my_model.predict(test_X)
survival = []
for i in predicted_survival:
    if i <= 0.5:
        survival.append(0)
    else:
        survival.append(1)

my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': survival})
my_submission.to_csv('submission.csv', index=False)