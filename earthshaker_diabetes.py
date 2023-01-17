import numpy as np

import pandas as pd

import math



import xgboost as xgb

from xgboost import XGBClassifier



from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
len(df_train)
labels = df_train['diabetes']
df_train.drop(columns=['p_id', 'diabetes'], axis=1, inplace=True)

df_test.drop(columns=['p_id'], axis=1, inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
#Scaling The Training Data

scaler = min_max_scaler.fit(df_train)

X_train_scaled = min_max_scaler.transform(df_train)

train = pd.DataFrame(X_train_scaled, columns=df_train.columns)

#Scaling The Testing Data

X_test_scaled = min_max_scaler.transform(df_test)

test = pd.DataFrame(X_test_scaled, columns=df_train.columns)
def xgb_CV(X, y, X_test, folds, params):

    

    prediction = np.zeros(len(X_test))

    scores = []

    

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

        

        model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, \

                          early_stopping_rounds=200, verbose_eval=150, params=params)

        

        y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        scores.append(accuracy_score(y_valid, y_pred_valid))

        

        prediction += y_pred

        

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

        

    prediction /= n_fold

    

    return prediction
params = {'eta': 0.01,

              'max_depth': 4,

              'subsample': 0.05,

              'colsample_bytree': 0.05,

              'min_child_weight' : 1,

              'objective': 'binary:hinge',

              'eval_metric': 'error',

              'silent': True,

              'nthread': 4}



n_fold = 10



folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)



predictions = xgb_CV(train, labels, test, folds, params)
preds = np.where(predictions > 0.5, 1, 0)



preds
df_submission = pd.read_csv('../input/sample_submission.csv')

df_submission['diabetes'] = preds

df_submission.to_csv('my_submission.csv', index=False)