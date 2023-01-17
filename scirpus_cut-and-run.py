import gc
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['target'] = np.log1p(train['target'])
test.insert(1,'target',-1.)
floatcolumns = []
intcolumns = []

for c in train.columns[2:]:
    s = train[c].dtype
    if(s=='float64'):
        floatcolumns.append(c)
    else:
        intcolumns.append(c)
alldata = pd.concat([train,test])
del train,test
gc.collect()
alldata['sumofzeros'] = (alldata[intcolumns]==0).sum(axis=1)
train = alldata[alldata.target!=-1]
test = alldata[alldata.target==-1]
del alldata
gc.collect()
x = pd.qcut(train.target, 7, labels=[0, 1, 2, 3, 4, 5, 6])
for a in [0, 1, 2, 3, 4, 5, 6]:
    print(a,train.target[x==a].min(),train.target[x==a].max())
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
    'zero_as_missing':True
}
feats = train.columns[2:]
# bestparams = {}
# for i in [0, 1, 2, 3, 4, 5]:
#     lgtrain = lgbm.Dataset(train[feats],x>i)
#     lgb_cv = lgbm.cv(
#         params = lgbm_params,
#         train_set = lgtrain,
#         num_boost_round=2000,
#         stratified=False,
#         nfold = 5,
#         verbose_eval=1,
#         seed = 42,
#         early_stopping_rounds=75)

#     optimal_rounds = np.argmin(lgb_cv['binary_logloss-mean'])
#     best_cv_score = min(lgb_cv['binary_logloss-mean'])
#     bestparams[i] = (optimal_rounds,best_cv_score)
#     del lgtrain
#     gc.collect()

bestparams = {0: (387, 0.32549223081879874),
             1: (363, 0.48964347828583543),
             2: (389, 0.5646998803516646),
             3: (412, 0.5738770523457457),
             4: (371, 0.5016286734255448),
             5: (433, 0.3353076747094824)}
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((train.shape[0],6))
sub_preds = np.zeros((test.shape[0],6))
for i in [0, 1, 2, 3, 4, 5]:
    optimal_rounds, best_cv_score = bestparams[i]
    print(i, optimal_rounds, best_cv_score)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
        print(n_fold)
        trn_x, trn_y = train[feats].iloc[trn_idx], x[trn_idx]>i
        val_x, val_y = train[feats].iloc[val_idx], x[val_idx]>i
        
        clf = lgbm.train(lgbm_params,
                         lgbm.Dataset(trn_x,trn_y),
                         num_boost_round = optimal_rounds + 1,
                         verbose_eval=200)

        oof_preds[val_idx,i] = clf.predict(val_x, num_iteration=optimal_rounds + 1)
        sub_preds[:,i] += clf.predict(test[feats], num_iteration=optimal_rounds + 1) / folds.n_splits

        del clf
        del trn_x, trn_y, val_x, val_y
        gc.collect()
off_preds_withbias = np.hstack([oof_preds,np.ones(shape=(oof_preds.shape[0],1))])
sub_preds_withbias = np.hstack([sub_preds,np.ones(shape=(sub_preds.shape[0],1))])
params = np.linalg.lstsq(off_preds_withbias, train.target)[0]
trainpreds = np.dot(off_preds_withbias,params)
print(np.sqrt(mean_squared_error(train.target,trainpreds)))
testpreds = np.dot(sub_preds_withbias,params)
sub = pd.DataFrame({'ID':test.ID, 'target':np.expm1(testpreds)})
sub.to_csv('cut.csv',index=False)