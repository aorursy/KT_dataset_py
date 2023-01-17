import gc

import numpy as np

import pandas as pd

import lightgbm as lgbm

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error
X = pd.read_csv('../input/andrews-new-stuff/train_features.csv')

X_test = pd.read_csv('../input/andrews-new-stuff/test_features.csv')

y = pd.read_csv('../input/andrews-new-stuff/y.csv')

submission = pd.read_csv('../input/andrews-new-stuff/submission.csv')



X['target'] = y.target

X_test['target'] = -1
x = pd.qcut(X.target, 7, labels=[0, 1, 2, 3, 4, 5, 6])
for a in [0, 1, 2, 3, 4, 5]:

    print(a,X.target[x==a].min(),X.target[x==a].max())
X.head()
X_test.head()
submission.head()
lgbm_params =  {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'binary_logloss',

    "learning_rate": 0.01,

    "num_leaves": 100,

    "feature_fraction": .5,

    "bagging_fraction": .5,

    #'bagging_freq': 4,

    "max_depth": -1,

    "reg_alpha": 0.3,

    "reg_lambda": 0.1,

    "min_child_weight":10,

    "n_jobs":4

}
feats = X.columns[:-1]
# bestparams = {}

# for i in [0, 1, 2, 3, 4, 5]:

#     lgtrain = lgbm.Dataset(X[feats],x>i)

#     lgb_cv = lgbm.cv(

#         params = lgbm_params,

#         train_set = lgtrain,

#         num_boost_round=2000,

#         stratified=False,

#         nfold = 5,

#         verbose_eval=0,

#         seed = 42,

#         early_stopping_rounds=75)



#     optimal_rounds = np.argmin(lgb_cv['binary_logloss-mean'])

#     best_cv_score = min(lgb_cv['binary_logloss-mean'])

#     bestparams[i] = (optimal_rounds,best_cv_score)

#     del lgtrain

#     gc.collect()

# bestparams
bestparams = {0: (343, 0.33300775159479323),

             1: (377, 0.3937832899637623),

             2: (381, 0.43248276195861485),

             3: (368, 0.4567765830007994),

             4: (279, 0.4364333328398183),

             5: (301, 0.3130615907253661)}
folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X.shape[0],6))

sub_preds = np.zeros((X_test.shape[0],6))
for i in [0, 1, 2, 3, 4, 5]:

    optimal_rounds, best_cv_score = bestparams[i]

    print(i, optimal_rounds, best_cv_score)

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X)):

        print(n_fold)

        trn_x, trn_y = X[feats].iloc[trn_idx], x[trn_idx]>i

        val_x, val_y = X[feats].iloc[val_idx], x[val_idx]>i

        

        clf = lgbm.train(lgbm_params,

                         lgbm.Dataset(trn_x,trn_y),

                         num_boost_round = optimal_rounds + 1,

                         verbose_eval=200)



        oof_preds[val_idx,i] = clf.predict(val_x, num_iteration=optimal_rounds + 1)

        sub_preds[:,i] += clf.predict(X_test[feats], num_iteration=optimal_rounds + 1) / folds.n_splits



        del clf

        del trn_x, trn_y, val_x, val_y

        gc.collect()
off_preds_withbias = np.hstack([oof_preds,np.ones(shape=(oof_preds.shape[0],1))])

sub_preds_withbias = np.hstack([sub_preds,np.ones(shape=(sub_preds.shape[0],1))])
params = np.linalg.lstsq(off_preds_withbias, y.target,rcond=-1)[0]
params
trainpreds = np.dot(off_preds_withbias,params)

print(mean_absolute_error(y.target,trainpreds))
testpreds = np.dot(sub_preds_withbias,params)

sub = pd.DataFrame({'seg_id':submission.seg_id, 'time_to_failure':testpreds})

sub.to_csv('cut.csv',index=False)
sub.time_to_failure.min()
sub.time_to_failure.max()