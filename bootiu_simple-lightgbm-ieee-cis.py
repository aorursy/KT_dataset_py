import os, time, gc, datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



import lightgbm as lgb



pd.set_option('max_rows', 9999)

pd.set_option('max_columns', 9999)



start = time.time()
# Data Loading

def load_data():

    train_tr = pd.read_csv('../input/train_transaction.csv')

    train_id = pd.read_csv('../input/train_identity.csv')

    test_tr = pd.read_csv('../input/test_transaction.csv')

    test_id = pd.read_csv('../input/test_identity.csv')

    

    train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

    test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

    del train_tr, train_id, test_tr, test_id

    gc.collect()

    

    return train, test
%%time

train, test = load_data()
train.head()
test.head()
print(train.shape)

print(test.shape)
# Object

obj_cols = train.select_dtypes(include=['object']).columns

for c in obj_cols:    

    train[c].fillna('Nodata', inplace=True)

    test[c].fillna('Nodata', inplace=True)

    

    lbl = LabelEncoder()

    lbl.fit(list(train[c].values) + list(test[c].values))

    train[c] = lbl.transform(train[c].values)

    test[c] = lbl.transform(test[c].values)



# float

num_cols = train.select_dtypes(include=['float']).columns

for c in num_cols:

    train[c] = train[c].astype(np.float32)

    test[c] = test[c].astype(np.float32)

sns.countplot(x='isFraud', data=train)

plt.show()
# make Dataset

features = [c for c in train.columns if c not in ['TransactionID', 'isFraud']]

target = 'isFraud'



X = train[features].values

Y = train[target].values



X_test = test[features].values



oof = np.zeros(len(train))

preds = np.zeros(len(test))

feature_importance = np.zeros(len(features))
# Config

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

params = {

    'task': 'train',

    'objective': 'binary',

    'metrics': 'auc',

    'max_depth': 9,

    'learning_rate': 0.005,

    'random_state': 0,

    'bagging_fraction': 0.9,

    'feature_fraction': 0.9,

}

config = {

    'num_boost_round': 12000,

    'early_stopping_rounds': 50,

    'verbose_eval': 2000

}
%%time

# Model training

for i, (trn_index, val_index) in enumerate(cv.split(X, Y)):

    print('{} Folds'.format(i + 1))

    

    _start = time.time()

    X_train, Y_train = X[trn_index], Y[trn_index]

    X_valid, Y_valid = X[val_index], Y[val_index]

    

    trn_data = lgb.Dataset(X_train, label=Y_train)

    val_data = lgb.Dataset(X_valid, label=Y_valid, reference=trn_data)

    

    model = lgb.train(params, trn_data, valid_sets=[val_data, trn_data], valid_names=['eval', 'train'], **config)

    

    oof[val_index] = model.predict(X_valid)

    preds += model.predict(X_test, iteration=model.best_iteration) / cv.get_n_splits()

    feature_importance += model.feature_importance(iteration=model.best_iteration)

    

    elapsedtime = time.time() - _start

    s = datetime.timedelta(seconds=elapsedtime)



    print('{} Folds Running Time: {}'.format(i + 1, str(s)))

    print('#' * 50)

    

    del model

    gc.collect()
# AUC

auc = roc_auc_score(train['isFraud'], oof)

print('oof AUC: {:.5f}'.format(auc))
importance_df = pd.DataFrame({

    'feature': features,

    'importance': feature_importance

})



fig = plt.figure(figsize=(12, 20))

sns.barplot(x='importance', y='feature', data=importance_df.sort_values(by='importance', ascending=False)[:50])

plt.show()
sub = pd.read_csv('../input/sample_submission.csv')

sub['isFraud'] = preds

sub.to_csv('submission.csv', index=False)
elapsedtime = time.time() - start

s = datetime.timedelta(seconds=elapsedtime)



print('This Kernel Running Time: {}'.format(str(s)))