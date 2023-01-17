import numpy as np

import pandas as pd

import multiprocessing

import sklearn

import lightgbm

from tqdm import tqdm
files = ['../input/lish-moa/test_features.csv', 

         '../input/lish-moa/train_targets_scored.csv',

         '../input/lish-moa/train_features.csv',

         '../input/lish-moa/train_targets_nonscored.csv',

         '../input/lish-moa/sample_submission.csv']



with multiprocessing.Pool() as pool:

    test, train_target, train, train_nonscored, submission = pool.map(pd.read_csv, files)
# train.head(3)

# test.head(3)

# train_target.head(3)

# sub.head(3)
sub=pd.get_dummies(train['cp_time'].map({24:0,48:1,72:2}),prefix='cp_time')

train=pd.concat([train,sub],axis=1)

sub=pd.get_dummies(test['cp_time'].map({24:0,48:1,72:2}),prefix='cp_time')

test=pd.concat([test,sub],axis=1)



for col in ['cp_type','cp_dose']:

    sub=pd.get_dummies(train[col])

    train=pd.concat([train,sub],axis=1)

    sub=pd.get_dummies(test[col])

    test=pd.concat([test,sub],axis=1)    

    

train=train.drop(['cp_type','cp_time','cp_dose'],axis=1)

test=test.drop(['cp_type','cp_time','cp_dose'],axis=1)
X_train=train.iloc[:,1:]

X_test=test.iloc[:,1:]

train_target=train_target.iloc[:,1:]
params={

      'max_bin':63,

      'device-type':'gpu',

      'num_leaves': 511,

      'feature_fraction': 0.3,

      'bagging_fraction': 0.3,

      'min_data_in_leaf': 100,

      'objective': 'binary',

      'max_depth': 9,

      'learning_rate': 0.05,

      'metric': 'binary_logloss',

      'verbosity': 0

}

cv=sklearn.model_selection.KFold(n_splits=2,shuffle=True,random_state=2020)



for col in tqdm(train_target.columns):

    y=train_target[col]

    y_preds=[]

    oof_train=np.zeros(X_train.shape[0])





    for train_idx,valid_idx in cv.split(X_train):

        X_tr,X_val=X_train.iloc[train_idx],X_train.iloc[valid_idx]

        y_tr,y_val=y.iloc[train_idx],y.iloc[valid_idx]



        lgb_train=lightgbm.Dataset(X_tr,y_tr)

        lgb_valid=lightgbm.Dataset(X_val,y_val,reference=lgb_train)



        model=lightgbm.train(params,lgb_train,num_boost_round=2000,valid_sets=[lgb_train,lgb_valid],

                        verbose_eval=0,early_stopping_rounds=50)



        oof_train[valid_idx]=model.predict(X_val,num_iteration=model.best_iteration)

    

        y_pred=model.predict(X_test,num_iteration=model.best_iteration)

        y_preds.append(y_pred)

        

    submission[col]=sum(y_preds)/len(y_preds)
submission.to_csv('submission.csv',index=False)