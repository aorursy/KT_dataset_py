import os



import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)



y_train = train['isFraud'].copy()

del train_transaction, train_identity, test_transaction, test_identity



# Drop target, fill in NaNs

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()



del train, test



# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))   

%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)

%%time

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

EPOCHS = 3

kf = KFold(n_splits = EPOCHS, shuffle = True)

y_preds = np.zeros(sample_submission.shape[0])

y_oof = np.zeros(X_train.shape[0])

for tr_idx, val_idx in kf.split(X_train, y_train):

    clf = xgb.XGBClassifier(

        n_estimators=1000,

        max_depth=15,

        learning_rate=0.08,

        subsample=0.9,

        colsample_bytree=0.9,

        tree_method='gpu_hist'

    )

    

    X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]

    y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    clf.fit(X_tr, y_tr)

    y_pred_train = clf.predict_proba(X_vl)[:,1]

    y_oof[val_idx] = y_pred_train

    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))

    

    y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS

 
sample_submission['isFraud'] = y_preds

sample_submission.to_csv('simple_xgboost.csv')