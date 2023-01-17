import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import xgboost as xgb  # gradient boosting library

from sklearn import preprocessing  # useful preprocessing utilities



import os

print(os.listdir('../input/rainfall-prediction-challenge'))  # check files in input directory
train = pd.read_csv('../input/rainfall-prediction-challenge/train.csv', index_col='row_id')

test = pd.read_csv('../input/rainfall-prediction-challenge/test.csv', index_col='row_id')



sample_submission = pd.read_csv('../input/rainfall-prediction-challenge/sample_submission.csv', index_col='row_id')



print(train.shape)

print(test.shape)
# drop target, fill in NaNs

y_train = train['rain_tomorrow'].copy()

X_train = train.drop('rain_tomorrow', axis=1)

X_test = test.copy()

X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)



# clean up memory

del train, test
# label encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))
# train xgboost classifier

clf = xgb.XGBClassifier(n_estimators=100,

                        max_depth=8,

                        learning_rate=0.1,

                        subsample=0.8,

                        colsample_bytree=0.8,

                        missing=-999,

                        verbose=2)



clf.fit(X_train, y_train)
# make submission file

predictions = clf.predict_proba(X_test)

sample_submission['rain_tomorrow'] = predictions[:,1]

sample_submission.to_csv('xgboost_baseline.csv')