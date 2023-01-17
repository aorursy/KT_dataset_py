import pandas as pd
import numpy as np

train = pd.read_csv('../input/otto-group-product-classification-challenge/train.csv')
test = pd.read_csv('../input/otto-group-product-classification-challenge/test.csv')
sampleSubmission = pd.read_csv('../input/otto-group-product-classification-challenge/sampleSubmission.csv')

train.shape, test.shape
train["target"] = train["target"].str.replace('Class_', '')
train["target"] = train["target"].astype(int) - 1

X_train = train.drop(['id','target'] , axis=1)
y_train = train["target"]
X_test = test.drop('id', axis=1)
y_pred = np.zeros((len(X_test), 9))
models = []
oof_train = np.zeros((len(X_train),9))
import lightgbm as lgb
from sklearn.model_selection import KFold

NFOLDS = 5

cv = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)

params = {
    #'metric':'multi_logloss',
    'objective': 'multiclass',
    'num_class': 9,
    #'verbosity': 1,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index].astype(int)
    y_val = y_train[valid_index].astype(int)
    
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_eval = lgb.Dataset(X_val, y_val)
    
    model = lgb.train(params, lgb_train,
                        valid_sets=[lgb_train, lgb_eval],
                        verbose_eval=10,
                        num_boost_round=1000,
                        early_stopping_rounds=10)
    
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred += model.predict(X_test, num_iteration=model.best_iteration)/NFOLDS
submit = pd.concat([sampleSubmission[['id']], pd.DataFrame(y_pred)], axis = 1)
submit.columns = sampleSubmission.columns
submit.to_csv('submit.csv', index=False)
submit