from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import preprocessing

#import xgboost as xgb

import numpy as np

import pandas as pd

import os

from sklearn.feature_selection import SelectPercentile, chi2

print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



test_ID = test_data.ID

target_data = train_data.target

target_for_sub = target_data.copy()

log_target_data = np.log1p(target_data)
test_data.head()
# this is clasification problem

# target_data.value_counts()

# le = preprocessing.LabelEncoder()

# y_train_encoded = le.fit_transform(target_data)

# y_test_encoded  = le.fit_transform(test_data)

# y_encoded
print('Test data: ', test_data.shape,'Target data: ', target_data.shape, 'Test data', train_data.shape)
# y = train_data['target']

# X = train_data.drop(['target', 'ID'], axis=1)

# X.head()
# no droped columns error

drop_train_rows = ['ID', 'target']

drop_test_rows = ['ID']







if set(drop_train_rows) & set(list(train_data.columns)):

    train_data.drop(drop_train_rows, axis=1, inplace=True)

if set(drop_test_rows) & set(list(test_data.columns)):

    test_data.drop(drop_test_rows, axis=1, inplace=True)

test_data.head()



# y = y.astype('float', inplace=True)

# y.unique
# new_train_data = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(train_data, 

                                                    log_target_data,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor()
regr.fit(X_train, y_train)

# lab_enc = preprocessing.LabelEncoder()

# training_scores_encoded = lab_enc.fit_transform(y_train)

# test_score_encoded = lab_enc.fit_transform(y_test)
predictions = regr.predict(X_test)

def rmsle(y_test, predictions):

    return np.sqrt(np.mean(np.power(y_test - predictions, 2)))

rmsle(y_test, predictions)

#pred_from_submissions = regr.predict(test_data)
X_train, X_test, y_train, y_test = train_test_split(train_data, 

                                                    target_data,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)

regr = RandomForestRegressor()

regr.fit(X_train, y_train)

pred_from_submissions = regr.predict(test_data)
sub = pd.DataFrame()

sub['ID'] = test_ID

sub['target'] = pred_from_submissions

sub.to_csv('submission.csv',index=False)
test = pd.read_csv('submission.csv')

test.head()