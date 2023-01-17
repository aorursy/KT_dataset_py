import numpy as np 

import pandas as pd 

import gc

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix

import xgboost as xgb
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.shape, test_df.shape
train_df.head()
train_cols = [c for c in train_df.columns if c not in ["ID_code", "target"]]

y_train = train_df["target"]
y_train.value_counts()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
# bayesian hpo

# details: https://xgboost.readthedocs.io/en/latest/parameter.html

params = {'tree_method': 'hist',

 'objective': 'binary:logistic',

 'eval_metric': 'auc',

 'learning_rate': 0.0936165921314771,

 'max_depth': 2,

 'colsample_bytree': 0.3561271102144279,

 'subsample': 0.8246604621518232,

 'min_child_weight': 53,

 'gamma': 9.943467991283027,

 'silent': 1}


%%time



oof_preds = np.zeros(train_df.shape[0])

sub_preds = np.zeros(test_df.shape[0])



feature_importance_df = pd.DataFrame()



for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_df, y_train)):

    

    trn_x, trn_y = train_df[train_cols].iloc[trn_idx], y_train.iloc[trn_idx]

    val_x, val_y = train_df[train_cols].iloc[val_idx], y_train.iloc[val_idx]

    

    dtrain = xgb.DMatrix(trn_x, trn_y, feature_names=trn_x.columns)

    dval = xgb.DMatrix(val_x, val_y, feature_names=val_x.columns)

    

    clf = xgb.train(params=params, dtrain=dtrain, num_boost_round=4000, evals=[(dtrain, "Train"), (dval, "Val")],

        verbose_eval= 100, early_stopping_rounds=50) 

    

    oof_preds[val_idx] = clf.predict(xgb.DMatrix(val_x))

    sub_preds += clf.predict(xgb.DMatrix(test_df[train_cols])) / folds.n_splits



    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    fold_importance_df["fold"] = n_fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    print('\nFold %1d AUC %.6f & std %.6f' %(n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx]), np.std([oof_preds[val_idx]])))

    print('Fold %1d Precision %.6f' %(n_fold + 1, precision_score(val_y, np.round(oof_preds[val_idx])) ))

    print('Fold %1d Recall %.6f' %(n_fold + 1, recall_score(val_y, np.round(oof_preds[val_idx]) )))

    print('Fold %1d F1 score %.6f' % (n_fold + 1,f1_score(val_y, np.round(oof_preds[val_idx]))))

    print('Fold %1d Kappa score %.6f\n' % (n_fold + 1,cohen_kappa_score(val_y, np.round(oof_preds[val_idx]))))

    gc.collect()



print('\nCV AUC score %.6f & std %.6f' % (roc_auc_score(y_train, oof_preds), np.std((oof_preds))))

print('CV Precision score %.6f' % (precision_score(y_train, np.round(oof_preds))))

print('CV Recall score %.6f' % (recall_score(y_train, np.round(oof_preds))))

print('CV F1 score %.6f' % (f1_score(y_train, np.round(oof_preds))))

print('CV Kappa score %.6f' % (cohen_kappa_score(y_train, np.round(oof_preds))))
print(confusion_matrix(y_train, np.round(oof_preds)))
fig, ax = plt.subplots(1,1,figsize=(10,12)) 

xgb.plot_importance(clf, max_num_features=20, ax=ax)  
fig, ax = plt.subplots(1,1,figsize=(10,12)) 

xgb.plot_importance(clf, max_num_features=20, ax=ax, importance_type="cover", xlabel="Cover")
fig, ax = plt.subplots(1,1,figsize=(10,12)) 

xgb.plot_importance(clf, max_num_features=20, ax=ax, importance_type="gain", xlabel="Gain")
feature_importance_df.groupby(["feature"])["fscore",].mean().sort_values("fscore", ascending=False)
test_df['target'] = sub_preds
test_df.head()
oof_roc = roc_auc_score(y_train, oof_preds)

oof_roc 
ss = pd.DataFrame({"ID_code":test_df["ID_code"], "target":test_df["target"]})

ss.to_csv("sant_xgb_%sFold_%.6f.csv"%(folds.n_splits, oof_roc), index=None)

ss.head()
ss.describe().T