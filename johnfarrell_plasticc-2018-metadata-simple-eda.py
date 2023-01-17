%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import pickle
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
DATA_DIR = '../input/'
# train = pd.read_csv(DATA_DIR+'training_set.csv')
# test_set_id = pd.read_csv(DATA_DIR+'test_set.csv', usecols=['object_id'])
# test_set_id.shape # (453653104, 1)
train = pd.read_csv(DATA_DIR+'training_set_metadata.csv')
test = pd.read_csv(DATA_DIR+'test_set_metadata.csv')
train.shape, test.shape
display(train.head())
display(test.head())
train.columns
feat_cols = [
    'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 
    'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 
    'distmod', 'mwebv'
]
len(feat_cols)
for c in feat_cols:
    print(
        'nan-ratio of {:>18} [train: {:.4f}; test: {:.4f}]'
        .format(
            c, 
            train[c].isnull().sum() / train.shape[0], 
            test[c].isnull().sum() / test.shape[0])
    )
for c in feat_cols:
    plt.figure(figsize=[20, 4])
    plt.subplot(1, 2, 1)
    sns.violinplot(x='target', y=c, data=train)
    plt.grid()
    plt.subplot(1, 2, 2)
    sns.distplot(train[c].dropna())
    sns.distplot(test[c].dropna())
    plt.legend(['train', 'test'])
    plt.grid()
    plt.show();
target = train['target'].values.copy()
del train['target']
train_ids = train['object_id'].copy()
test_ids = test['object_id'].copy()
del train['object_id'], test['object_id'];
train['target'] = target.copy()
sns.pairplot(train[feat_cols+['target']].dropna(), hue='target', vars=feat_cols)
del train['target']
data = pd.concat([
    train[feat_cols], 
    test[feat_cols].sample(frac=5 * train.shape[0]/test.shape[0])
], ignore_index=True)
data['is_test'] = 1
data['is_test'][:train.shape[0]] = 0
sns.pairplot(data, hue='is_test', vars=feat_cols, plot_kws={'alpha': 0.25})
del data; gc.collect();
tmp = False
for i in [6, 16, 53, 65, 92]:
    tmp|=(target==i)
print(
    tmp.sum(), 
    (train[['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err']].sum(1)==0).sum(), 
    (train['distmod'].isnull()).sum()
)
print(
    (test[[
        'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err'
    ]].sum(1)==0).sum(), (test['distmod'].isnull()).sum()
)
train_mask = train['distmod'].isnull().values
test_mask = test['distmod'].isnull().values
labels2weight = {x:1 for x in np.unique(target)}
labels2weight[64] = 2
labels2weight[15] = 2
import lightgbm as lgb

round_params = dict(num_boost_round = 20000,
                    early_stopping_rounds = 100,
                    verbose_eval = 50)
params = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    #"num_class": len(np.unique(y)),
    #"two_round": True,
    "num_leaves" : 30,
    "min_child_samples" : 30,
    "learning_rate" : 0.03,
    "feature_fraction" : 0.75,
    "bagging_fraction" : 0.75,
    "bagging_freq" : 1,
    "seed" : 42,
    "lambda_l2": 1e-2,
    "verbosity" : -1
}

def lgb_cv_train(X, labels, X_test, 
                 params=params, round_params=round_params):
    print('X', X.shape, 'labels', labels.shape, 'X_test', X_test.shape)
    print('unique labels', np.unique(labels))
    
    labels2y = dict(map(reversed, enumerate(np.unique(labels))))
    y2labels = dict(enumerate(np.unique(labels)))
    y = np.array(list(map(labels2y.get, labels)))
    weight = np.array(list(map(labels2weight.get, labels)))
    
    params['num_class'] = len(np.unique(y))
    cv_raw = lgb.cv(
        params, 
        lgb.Dataset(X, label=y, weight=weight), 
        nfold=5, 
        **round_params
    )
    best_round = np.argmin(cv_raw['multi_logloss-mean'])
    best_score = cv_raw['multi_logloss-mean'][best_round]
    print(f'best_round: {best_round}', f'best_score: {best_score}')
    model = lgb.train(
        params, 
        lgb.Dataset(X, label=y, weight=weight), 
        num_boost_round=best_round, 
    )
    pred = model.predict(X_test)
    pred_labels = pd.DataFrame(
        {f'class_{c}': pred[:, i] for i,c in enumerate(np.unique(labels))}
    )
    res = dict(
        model=model,
        best_round=best_round,
        best_score=best_score,
        pred_labels=pred_labels
    )
    return res
feat_extra_li = ['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod']
feat_gal_cols = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'mwebv']
feat_extra_cols = feat_gal_cols + feat_extra_li
print(feat_gal_cols)
print(feat_extra_cols)
np.unique(target[train_mask]), np.unique(target[~train_mask])
%%time
res_gal = lgb_cv_train(
    train.loc[train_mask, feat_gal_cols], 
    target[train_mask], 
    test.loc[test_mask, feat_gal_cols]
)
res_gal['pred_labels'].head()
n_gal = res_gal['pred_labels'].shape[1]
res_gal['pred_labels'] = res_gal['pred_labels'] * n_gal/(n_gal+1)
res_gal['pred_labels']['class_99'] = 1/(n_gal+1)
res_gal['pred_labels'].head()
%%time
res_extra = lgb_cv_train(
    train.loc[~train_mask, feat_extra_cols], 
    target[~train_mask], 
    test.loc[~test_mask, feat_extra_cols]
)
res_extra['pred_labels'].head()
n_extra = res_extra['pred_labels'].shape[1]
res_extra['pred_labels'] = res_extra['pred_labels'] * n_extra/(n_extra+1)
res_extra['pred_labels']['class_99'] = 1/(n_extra+1)
res_extra['pred_labels'].head()
sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
sub = sub.set_index('object_id')
sub[:] = 0
sub.head()
classnames = sub.columns.tolist()
print(sub.shape, classnames)
for c in res_gal['pred_labels'].columns:
    sub.loc[test_mask, c] = res_gal['pred_labels'][c].values
for c in res_extra['pred_labels'].columns:
    sub.loc[~test_mask, c] = res_extra['pred_labels'][c].values
sub.tail(10)
%%time
score = res_gal['best_score'] * (train_mask).sum()/train.shape[0]
score+= res_extra['best_score'] * (~train_mask).sum()/train.shape[0]
sub.reset_index().to_csv(f'meta_lgb_{score}.csv', index=False, float_format='%.6f')
os.listdir()
train['is_train'] = 1
test['is_train'] = 0
X = pd.concat(
    [train, 
     test
     .sample(frac=5*train.shape[0]/test.shape[0])],
    ignore_index=True
)
y = X['is_train'].values.copy()
del X['is_train'], train['is_train'], test['is_train']
del X['hostgal_specz'] # this is obvious different disttributed in train/test and will make auc=0.99
X.head()
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
params['objective'] = 'binary'
params['metric'] = 'auc'
params['num_class'] = 1

print('X', X.shape, 'y', y.shape)
pred_adv = np.zeros(X.shape[0])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
dtrain = lgb.Dataset(X, label=y)
dtrain.construct()
models = []
for trn_idx, val_idx in kf.split(X):
    model = lgb.train(
        params, 
        dtrain.subset(trn_idx), 
        valid_sets=[dtrain.subset(trn_idx), dtrain.subset(val_idx)], 
        valid_names=['train', 'valid'],
        **round_params
    )
    models.append(model)
    pred_adv[val_idx] = model.predict(X.iloc[val_idx])
adv_score = roc_auc_score(y, pred_adv)
print('oof score:', adv_score)
def get_imp_plot(name, feature_name, lgb_feat_imps, nfolds=5, savefig=False):
    lgb_imps = pd.DataFrame(
        np.vstack(lgb_feat_imps).T, 
        columns=['fold_{}'.format(i) for i in range(nfolds)],
        index=feature_name,
    )
    lgb_imps['fold_mean'] = lgb_imps.mean(1)
    lgb_imps = lgb_imps.loc[
        lgb_imps['fold_mean'].sort_values(ascending=False).index
    ]
    lgb_imps.reset_index().to_csv(f'{name}_lgb_imps.csv', index=False)
    del lgb_imps['fold_mean']; gc.collect();

    max_num_features = min(len(feature_name), 300)
    f, ax = plt.subplots(figsize=[8, max_num_features//2])
    data = lgb_imps.iloc[:max_num_features].copy()
    data_mean = data.mean(1).sort_values(ascending=False)
    data = data.loc[data_mean.index]
    data_index = data.index.copy()
    data = [data[c].values for c in data.columns]
    data = np.hstack(data)
    data = pd.DataFrame(data, index=data_index.tolist()*nfolds, columns=['igb_imp'])
    data = data.reset_index()
    data.columns = ['feature_name', 'igb_imp']
    sns.barplot(x='igb_imp', y='feature_name', data=data, orient='h', ax=ax)
    plt.grid()
    if savefig:
        plt.savefig(f'{name}_lgb_imp.png')

get_imp_plot(
    'adv', 
    X.columns.tolist(), 
    [model.feature_importance()/model.best_iteration for model in models]
)