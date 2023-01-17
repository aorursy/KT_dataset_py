import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import gc

%matplotlib inline

print(os.listdir("../input"))

train_meta = pd.read_csv('../input/training_set_metadata.csv')
test_meta = pd.read_csv('../input/test_set_metadata.csv')
# Make sure I can rerun that any time
if 'object_id' in train_meta:
    del train_meta['object_id'], train_meta['target']
    del test_meta['object_id']
    gc.collect()
    
train_meta['is_train'] = 1
test_meta['is_train'] = 0
np.random.seed(10)
full_meta = pd.concat([train_meta, test_meta], axis=0).sample(frac=1.0)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

features = [f for f in full_meta if f not in ['is_train']]
importances = pd.DataFrame()

lgb_params = {
    'n_estimators': 200,
    'boosting_type': 'rf',
    'learning_rate': .05,
    'num_leaves': 127,
    'subsample': 0.621,
    'colsample_bytree': 0.7,
    'max_depth': 7,
    'bagging_freq': 1,
}
oof_preds = np.zeros(full_meta.shape[0])
for fold_, (trn_, val_) in enumerate(folds.split(full_meta, full_meta['is_train'])):
    trn_x, trn_y = full_meta[features].iloc[trn_], full_meta['is_train'].iloc[trn_]
    val_x, val_y = full_meta[features].iloc[val_], full_meta['is_train'].iloc[val_]
    
    clf = lgb.LGBMClassifier(**lgb_params)
    
    clf.fit(trn_x, trn_y)
    
    oof_preds[val_] = clf.predict_proba(val_x)[:, 1]
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = features
    imp_df['gain'] = clf.booster_.feature_importance(importance_type='gain')
    imp_df['split'] = clf.booster_.feature_importance(importance_type='split')
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    del trn_x, trn_y, val_x, val_y
    gc.collect()

print('Train/Test separation score : %.6f' % roc_auc_score(full_meta['is_train'], oof_preds))
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
plt.tight_layout()
nulls = pd.concat([train_meta.isnull().mean(), test_meta.isnull().mean()], axis=1).sort_index()
nulls.columns = ['train', 'test']
nulls

del train_meta, test_meta, full_meta
gc.collect()
train = pd.read_csv('../input/training_set.csv')
train.shape
# Drop object_id
del train['object_id']
train['is_train'] = 1
import time
start = time.time()
chunks = 5000000
importances = pd.DataFrame()
for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):
    # Drop object_id
    del df['object_id']
    # Add is_train feature
    df['is_train'] = 0
    # Concat
    full_data = pd.concat([df, train], axis=0).sample(frac=1.0)
    y = full_data['is_train']
    del full_data['is_train'], df
    gc.collect()
    
    # Run a lightgbm
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    
    oof_preds = np.zeros(full_data.shape[0])
    for fold_, (trn_, val_) in enumerate(folds.split(full_data, y)):
        trn_x, trn_y = full_data.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_data.iloc[val_], y.iloc[val_]

        clf = lgb.LGBMClassifier(**lgb_params)

        clf.fit(trn_x, trn_y)

        oof_preds[val_] = clf.predict_proba(val_x)[:, 1]

        imp_df = pd.DataFrame()
        imp_df['feature'] = full_data.columns
        imp_df['gain'] = clf.booster_.feature_importance(importance_type='gain')
        imp_df['split'] = clf.booster_.feature_importance(importance_type='split')
        imp_df['fold'] = i_c * folds.n_splits + fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        del trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Train/Test separation score : %.6f' % roc_auc_score(y, oof_preds))
    
    print('Chunk %3d Adversarial validation done [%5.1fmin spent]' % (i_c+1, (time.time() - start)/60))
    
    del full_data
    gc.collect()
    
    if i_c > 4 :
        break
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
plt.tight_layout()
