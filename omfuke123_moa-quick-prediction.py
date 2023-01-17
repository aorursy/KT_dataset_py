import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
train_features.drop(['sig_id','cp_type','cp_dose'],axis=1,inplace=True)
targets = [x for x in train_targets_scored.columns if x!='sig_id']
test_features.drop(['sig_id','cp_type','cp_dose'],axis=1,inplace=True)
params = {}
params["boosting_type"]= "gbdt",
params["objective"] = "binary"
params['metric'] = "binary_logloss"
params["learning_rate"] = 0.05
params["min_child_weight"] = 1
params["bagging_fraction"] = 0.8
params["bagging_seed"] = 2017
params["feature_fraction"] = 0.7
params["verbosity"] = 0
params["max_depth"] = 20
params["nthread"] = -1
skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
total_loss = 0
for model,target in enumerate(targets,1):
    y = train_targets_scored[target]
    predictions = np.zeros(test_features.shape[0])
    oof_preds = np.zeros(train_features.shape[0])
    
    for train_idx, test_idx in skf.split(train_features, y):
        train_data = lgb.Dataset(train_features.iloc[train_idx], label=y.iloc[train_idx])
        val_data = lgb.Dataset(train_features.iloc[test_idx], label=y.iloc[test_idx])
        clf = lgb.train(params, train_data, 1000, valid_sets = [train_data, val_data], verbose_eval=0, early_stopping_rounds=30)
        oof_preds[test_idx] = clf.predict(train_features.iloc[test_idx])
        predictions += clf.predict(test_features) / skf.n_splits
        
    ss[target] = predictions
    loss = log_loss(y, oof_preds)
    total_loss += loss
    
    print(f"Model:{model} ==> Losses:{loss:.4f}")
ss.to_csv('submission.csv',index=False)