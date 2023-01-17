%matplotlib inline
import os
import gc
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
DATA_DIR = '../input/'
train_meta = pd.read_csv(DATA_DIR + '/training_set_metadata.csv')
test_meta = pd.read_csv(DATA_DIR + '/test_set_metadata.csv')
display(train_meta.shape)
display(train_meta.head())
display(test_meta.shape)
display(test_meta.head())
labels = np.hstack([np.unique(train_meta['target']), [99]])
target_map = {j:i for i, j in enumerate(labels)}
target_ids = [target_map[i] for i in train_meta['target']]
train_meta['target_id'] = target_ids

# Build the flat probability arrays for both the galactic and extragalactic groups
galactic_cut = train_meta['hostgal_photoz'] == 0
galactic_data = train_meta[galactic_cut]
extragalactic_data = train_meta[~galactic_cut]

galactic_classes = np.unique(galactic_data['target_id'])
extragalactic_classes = np.unique(extragalactic_data['target_id'])

print('galactic_classes = ', galactic_classes)
print('extragalactic_classes = ', extragalactic_classes)

# Add class 99 (target_id = 14) to both groups.
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)

galactic_probabilities = np.zeros(15)
galactic_probabilities[galactic_classes] = 1. / len(galactic_classes)
extragalactic_probabilities = np.zeros(15)
extragalactic_probabilities[extragalactic_classes] = 1. / len(extragalactic_classes)
# Apply this naive prediction to a table
def do_prediction(table):
    probs = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
        else:
            prob = extragalactic_probabilities
        probs.append(prob)
    return np.array(probs)

pred = do_prediction(train_meta)
test_pred = do_prediction(test_meta)
train_pred = pd.DataFrame(index=train_meta['object_id'], data=pred, columns=['class_%d' % i for i in labels])
test_pred = pd.DataFrame(index=test_meta['object_id'], data=test_pred, columns=['class_%d' % i for i in labels])
naive_path = 'naive.csv'
train_pred.to_csv('train_' + naive_path)
test_pred.to_csv(naive_path)
target = train_meta['target'].values.copy()
del train_meta['target']
train_ids = train_meta['object_id'].copy()
test_ids = test_meta['object_id'].copy()
del train_meta['object_id'], test_meta['object_id']
labels2weights = {x:1 for x in np.unique(target)}
labels2weights[15] = 2
labels2weights[64] = 2
train_mask = train_meta['hostgal_photoz'] == 0
test_mask = test_meta['hostgal_photoz'] == 0
sum(train_mask), sum(test_mask)
round_params = dict(num_boost_round = 20000,
                    early_stopping_rounds = 200,
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

def lgb_cv_train(X, target, X_test, 
                 params=params, round_params=round_params):
    print('X', X.shape, 'target', target.shape, 'X_test', X_test.shape)
    labels = np.unique(target)
    print('unique labels', labels)
    
    ys2labels = dict(enumerate(labels))
    labels2ys = dict(map(reversed, enumerate(labels)))
    ys = np.array(list(map(labels2ys.get, target)))
    weights = np.array(list(map(labels2weights.get, target)))
    
    params['num_class'] = len(labels)
    cv_raw = lgb.cv(
        params, 
        lgb.Dataset(X, label=ys, weight=weights),
        nfold=5, 
        **round_params
    )
    best_round = np.argmin(cv_raw['multi_logloss-mean'])
    best_score = cv_raw['multi_logloss-mean'][best_round]
    print(f'best_round: {best_round}', f'best_score: {best_score}')
    model = lgb.train(
        params, 
        lgb.Dataset(X, label=ys, weight=weights), 
        num_boost_round=best_round, 
    )
    pred = model.predict(X_test)
    pred_labels = pd.DataFrame(
        {f'class_{c}': pred[:, i] for i,c in enumerate(labels)}
    )
    pred_train = model.predict(X)
    pred_train_labels = pd.DataFrame(
        {f'class_{c}': pred_train[:, i] for i,c in enumerate(labels)}
    )
    res = dict(
        model=model,
        best_round=best_round,
        best_score=best_score,
        pred_labels=pred_labels,
        pred_train_labels=pred_train_labels
    )
    return res
feat_extra_li = ['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod']
feat_gal_cols = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'mwebv']
feat_extra_cols = feat_gal_cols + feat_extra_li
np.unique(target[train_mask]), np.unique(target[~train_mask])
%%time
res_gal = lgb_cv_train(
    train_meta.loc[train_mask, feat_gal_cols], 
    target[train_mask], 
    test_meta.loc[test_mask, feat_gal_cols]
)
n_gal = res_gal['pred_labels'].shape[1]
res_gal['pred_labels'] = res_gal['pred_labels'] * n_gal/(n_gal+1)
res_gal['pred_labels']['class_99'] = 1/(n_gal+1)
# res_gal['pred_labels'].head()
%%time
res_extra = lgb_cv_train(
    train_meta.loc[~train_mask, feat_extra_cols], 
    target[~train_mask], 
    test_meta.loc[~test_mask, feat_extra_cols]
)
n_extra = res_extra['pred_labels'].shape[1]
res_extra['pred_labels'] = res_extra['pred_labels'] * n_extra/(n_extra+1)
res_extra['pred_labels']['class_99'] = 1/(n_extra+1)
# res_extra['pred_labels'].head()
train_pred = pd.DataFrame(index=train_ids, columns=['class_%d' % i for i in labels])
train_pred[:] = 0
for c in res_gal['pred_train_labels'].columns:
    train_pred.loc[train_mask.values, c] = res_gal['pred_train_labels'][c].values
for c in res_extra['pred_train_labels'].columns:
    train_pred.loc[~train_mask.values, c] = res_extra['pred_train_labels'][c].values
lgb_meta_path = 'lgb_meta.csv'
train_pred.to_csv('train_' + lgb_meta_path)
test_pred = pd.DataFrame(index=test_ids, columns=['class_%d' % i for i in labels])
test_pred[:] = 0
for c in res_gal['pred_labels'].columns:
    test_pred.loc[test_mask.values, c] = res_gal['pred_labels'][c].values
for c in res_extra['pred_labels'].columns:
    test_pred.loc[~test_mask.values, c] = res_extra['pred_labels'][c].values
test_pred.to_csv(lgb_meta_path)
from sklearn.metrics import log_loss

weights = np.array(list(map(labels2weights.get, target)))
labels = list(labels2weights.keys()) + [99]

naive = pd.read_csv('train_' + naive_path)
del naive['object_id']
score_naive = log_loss(target, naive, eps=1e-15, normalize=True, sample_weight=weights, labels=labels)
lgb_meta = pd.read_csv('train_' + lgb_meta_path)
del lgb_meta['object_id']
score_lgb_meta = log_loss(target, lgb_meta, eps=1e-15, normalize=True, sample_weight=weights, labels=labels)

print('score_naive_sklearn = ', score_naive)
print('score_lgb_meta_sklearn = ', score_lgb_meta)
def to_df(target):
    target_df = pd.DataFrame(index=train_ids, columns=['class_%d' % i for i in labels])
    target_df[:] = 0
    for i, y in enumerate(target):
        target_df.iloc[i]['class_%d' % y] = 1
    return target_df

def score(target, y_pred, weights, eps=1e-15):
    targets_df = to_df(target)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    sum_by_class = -(targets_df * np.log(y_pred)).sum(axis=0)
    num_in_class =  np.clip(targets_df.sum(axis=0), 1, float('inf'))
    score = (weights * sum_by_class / num_in_class).sum(axis=1) / weights.sum(axis=1)
    return float(score)

class_weights = {'class_%d' % k: labels2weights[k] for k in labels2weights}
weights = pd.DataFrame(class_weights, index=[0])

naive = pd.read_csv('train_' + naive_path).set_index('object_id')
print('score_naive = ', score(target, naive, weights))

lgb_meta = pd.read_csv('train_' + lgb_meta_path).set_index('object_id')
print('score_lgb_meta = ', score(target, lgb_meta, weights))
# !pip install kaggle
# path = '/tmp/.kaggle/'
# os.environ['KAGGLE_CONFIG_DIR'] = path
# basedir = os.path.dirname(path)
# if not os.path.exists(basedir):
#     os.makedirs(basedir)
# filename = path + 'kaggle.json'    
# if not os.path.isfile(filename):
#     with open(filename, 'a') as f:
#         f.write('{"username":"your name","key":"your kaggle key"}')
# !chmod 600 {filename}
# print('$HOME=', os.environ['HOME'])
# print('$KAGGLE_CONFIG_DIR', os.environ['KAGGLE_CONFIG_DIR'])
# !kaggle competitions submit -f {naive_benchmark_path} -m naive_benchmark PLAsTiCC-2018
# !kaggle competitions submit -f {lgb_meta_path} -m lgb_meta PLAsTiCC-2018
# submissions = !kaggle competitions submissions -csv PLAsTiCC-2018
# header = submissions[0].split()
# naive_submit = [s for s in submissions[2:] if s.startswith(naive_benchmark_path)]
# lgb_meta_submit = [s for s in submissions[2:] if s.startswith(lgb_meta_path)]
# score_naive_public = naive_submit[0].split()[-2]
# score_lgb_meta_public = lgb_meta_submit[0].split()[-2]
# print('score_naive_public = ', score_naive_public)
# print('score_lgb_meta_public = ', score_lgb_meta_public)