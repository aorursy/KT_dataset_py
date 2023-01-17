import os

import warnings

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
def encodeLabels(X_train, X_test):

    # Label Encoding

    for f in X_train.columns:

        if X_train[f].dtype=='object' or X_test[f].dtype=='object':

            lbl = LabelEncoder()

            lbl.fit(list(X_train[f].values) + list(X_test[f].values))

            X_train[f] = lbl.transform(list(X_train[f].values))

            X_test[f] = lbl.transform(list(X_test[f].values))

    return X_train, X_test



def readData(path, encode=True):

    """

    Read train/test data

    Parameters:

        1. path: input path to the train/test csv files (String)

        2. encode: weather to label encode categorical columns (Boolean)

    Outputs:

        1. X_train:

        2. y_train:

        3. X_test:

        4. y_test:

    """

    # Loading train/test data

    train_transaction = pd.read_csv(os.path.join(path, 'train_transaction.csv'), index_col='TransactionID')

    train_identity = pd.read_csv(os.path.join(path, 'train_identity.csv'), index_col='TransactionID')

    test_transaction = pd.read_csv(os.path.join(path, 'test_transaction.csv'), index_col='TransactionID')

    test_identity = pd.read_csv(os.path.join(path, 'test_identity.csv'), index_col='TransactionID')

    sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'), index_col='TransactionID')



    # Merging the transaction and identity

    train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

    test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



    # Creating dataframes

    X_train = train.drop('isFraud', axis=1)

    y_train = train['isFraud'].copy()

    X_test = test.copy()

    y_test = sample_submission.copy()



    if encode==True:

        X_train, X_test = encodeLabels(X_train, X_test)



    return X_train, y_train, X_test, y_test
X, y, test, submission = readData("../input/")
print("Percentage of fraud records = {:.2f}%".format((y[y==1].shape[0]/y.shape[0])*100))
nsplits = 5

submission["isFraud"] = 0

skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=0)



parameters = {

    'objective': 'binary',

    'metric': 'auc',

    'learning_rate': 0.05,

    'n_estimators': 500,

    'max_depth': 9,

    'bagging_fraction': 0.9,

    'feature_fraction': 0.9,

}



for idx_train, idx_test in skf.split(X, y):

    train_data = lgb.Dataset(data=X.iloc[idx_train], label=y.iloc[idx_train])

    valid_data = lgb.Dataset(data=X.iloc[idx_test], label=y.iloc[idx_test])

    model = lgb.train(params=parameters, train_set=train_data, valid_sets=valid_data, \

                      verbose_eval=500, early_stopping_rounds=100)

    submission['isFraud'] = submission['isFraud'] + model.predict(test)

    

submission['isFraud'] = submission['isFraud'] / 5
submission.to_csv("submission.csv")