import pandas as pd

import xgboost as xgb

from sklearn import preprocessing

import gc
%%time



train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

train_transaction = pd.read_csv('../input/ieee-fraud-detection//train_transaction.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)



y_train = train['isFraud'].copy()

del train_identity, train_transaction, test_identity, test_transaction



X_train = train.drop('isFraud', axis=1)

X_test = test.copy()



del train, test



X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)



for col in X_train.columns:

    if X_train[col].dtype == 'object' or X_test[col].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[col].values) + list(X_test[col].values))

        X_train[col] = lbl.transform(list(X_train[col].values))

        X_test[col] = lbl.transform(list(X_test[col].values))
clf = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=9,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    missing=-999,

    random_state=2000

)
%time clf.fit(X_train, y_train)
sample_submission['isFraud'] = clf.predict_proba(X_test)[:, 1]

sample_submission.to_csv('xgboost.csv')