import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
from itertools import product
import re

# import torch
# import torch.nn.functional as F
# from torch.autograd import grad
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

import pdb
import gc

sns.set_style('whitegrid')
# # this is a reimplementation of the above loss function using pytorch expressions.
# # Alternatively this can be done in pure numpy (not important here)
# # note that this function takes raw output instead of probabilities from the booster
# # Also be aware of the index order in LightDBM when reshaping (see LightGBM docs 'fobj')
# def wloss_metric(preds, train_data):
#     y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
#     y_h = torch.zeros(
#         y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
#     y_h /= y_h.sum(dim=0, keepdim=True)
#     y_p = torch.tensor(preds, requires_grad=False).type(torch.FloatTensor)
#     if len(y_p.shape) == 1:
#         y_p = y_p.reshape(len(classes), -1).transpose(0, 1)
#     ln_p = torch.log_softmax(y_p, dim=1)
#     wll = torch.sum(y_h * ln_p, dim=0)
#     loss = -torch.dot(weight_tensor, wll) / torch.sum(weight_tensor)
#     return 'wloss', loss.numpy() * 1., False


# # with autograd or pytorch you can pretty much come up with any loss function you want
# # without worrying about implementing the gradients yourself
# def wloss_objective(preds, train_data):
#     y_t = torch.tensor(train_data.get_label(), requires_grad=False).type(torch.LongTensor)
#     y_h = torch.zeros(
#         y_t.shape[0], len(classes), requires_grad=False).scatter(1, y_t.reshape(-1, 1), 1)
#     ys = y_h.sum(dim=0, keepdim=True)
#     y_h /= ys
#     y_p = torch.tensor(preds, requires_grad=True).type(torch.FloatTensor)
#     y_r = y_p.reshape(len(classes), -1).transpose(0, 1)
#     ln_p = torch.log_softmax(y_r, dim=1)
#     wll = torch.sum(y_h * ln_p, dim=0)
#     loss = -torch.dot(weight_tensor, wll)
#     grads = grad(loss, y_p, create_graph=True)[0]
#     grads *= float(len(classes)) / torch.sum(1 / ys)  # scale up grads
#     hess = torch.ones(y_p.shape)  # haven't bothered with properly doing hessian yet
#     return grads.detach().numpy(), \
#         hess.detach().numpy()
classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1,
                64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
weight_tensor = tf.convert_to_tensor(list(class_weight.values()), dtype=tf.float32)
class_dict = {c: i for i, c in enumerate(classes)}

def label_to_code(labels):
    return np.array([class_dict[c] for c in labels])

# this is the simplified original loss function by Olivier. It works excellently as an
# evaluation function, but we won't be able to use it in training
def multi_weighted_logloss(y_true, y_preds):
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    enc = OneHotEncoder(sparse=False, categories='auto')
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, 1)
    y_ohe = enc.fit_transform(y_true)
    y_p = np.clip(a=y_preds, a_min=1e-15, a_max=1 - 1e-15)
    if y_p.shape[0] > y_true.shape[0]:
        y_p = y_p.reshape(y_true.shape[0], len(classes), order='F')
        if y_p.shape[0] != y_true.shape[0]:
            raise ValueError(
                'Dimension Mismatch for y_p {0} and y_true {1}!'.format(
                    y_p.shape, y_true.shape))
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(np.multiply(y_ohe, y_p_log), axis=0)
    nb_pos = np.sum(y_ohe, axis=0).astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

# we may implement the above loss function using Tensorflow then use automatic 
# differentiation for the grad and hess in the objective function
def wloss_metric(preds, train_data):
    y_t = tf.convert_to_tensor(train_data.get_label())
    y_h = tf.one_hot(y_t, depth=14, dtype=tf.float32)
    y_h /= tf.reduce_sum(y_h, axis=0, keepdims=True)
    y_p = tf.convert_to_tensor(preds, dtype=tf.float32)
    if len(y_p.shape) == 1:
        y_p = tf.transpose(tf.reshape(y_p, (len(classes), -1)), perm=(1, 0))
#     ln_p = tf.nn.log_softmax(y_p, axis=1)
    ln_p = tf.log(tf.clip_by_value(tf.nn.softmax(y_p, axis=1), 1e-15, 1-1e-15))
    wll = tf.reduce_sum(y_h * ln_p, axis=0)
    loss = -tf.reduce_sum(weight_tensor * wll) / tf.reduce_sum(weight_tensor)
    return 'wloss', loss.numpy(), False

def grad(f):
    return lambda x: tfe.gradients_function(f)(x)[0]

def wloss_objective(preds, train_data):
    y_t = tf.convert_to_tensor(train_data.get_label())
    y_h = tf.one_hot(y_t, depth=14, dtype=tf.float32)
    ys = tf.reduce_sum(y_h, axis=0, keepdims=True)
    y_h /= ys
    y_p = tf.convert_to_tensor(preds, dtype=tf.float32)
    def loss(y_p):
        if len(y_p.shape) == 1:
            y_p = tf.transpose(tf.reshape(y_p, (len(classes), -1)), perm=(1, 0))
        ln_p = tf.nn.log_softmax(y_p, axis=1)
#         ln_p = tf.log(tf.clip_by_value(tf.nn.softmax(y_p, axis=1), 1e-15, 1-1e-15))
        wll = tf.reduce_sum(y_h * ln_p, axis=0)
        return -tf.reduce_sum(weight_tensor * wll) * len(train_data.get_label())
    grads = grad(loss)(y_p)
#     hess = grad(grad(loss))(y_p)
#     hess /= tf.reduce_mean(hess)
    hess = tf.ones(y_p.shape)
    return grads.numpy(), hess.numpy()

def softmax(x, axis=1):
    z = np.exp(x)
    return z / np.sum(z, axis=axis, keepdims=True)
# we use some synthetic data to verify that the loss functions are correct:
mock_y_true = np.array(classes + [6] * (100 - 14))
mock_pred_score = np.zeros((100, 14))
mock_pred_score[:, 0] = 10
mock_pred_score[:, 1:] = 5
mock_preds = softmax(mock_pred_score)
multi_weighted_logloss(mock_y_true, mock_preds)
wloss_metric(np.reshape(mock_pred_score, (-1), order='F'),
             lgb.Dataset(None, label_to_code(mock_y_true)))[1]
train_meta = pd.read_csv('../input/training_set_metadata.csv')
gc.enable()

train = pd.read_csv('../input/training_set.csv')
train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']

aggs = {
#     'mjd': ['min', 'max', 'size'],
#     'passband': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean','std'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
}

train_feats = train.groupby('object_id').agg(aggs)
new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]
train_feats.columns = new_columns
detected_groups = train[train['detected'] == 1].groupby('object_id')
train_feats['det_mjd_diff'] = detected_groups['mjd'].transform('max') \
    - detected_groups['mjd'].transform('min')
train_feats['flux_diff'] = train_feats['flux_max'] - train_feats['flux_min']
train_feats['flux_dif2'] = (train_feats['flux_max'] - 
                            train_feats['flux_min']) / train_feats['flux_mean']
train_feats['flux_w_mean'] = train_feats['flux_by_flux_ratio_sq_sum'] \
    / train_feats['flux_ratio_sq_sum']
train_feats['flux_dif3'] = (train_feats['flux_max'] - 
                            train_feats['flux_min']) / train_feats['flux_w_mean']

# del train_feats['mjd_max'], train_feats['mjd_min']

del train
gc.collect()
full_features = train_feats.reset_index().merge(
    right=train_meta,
    how='outer',
    on='object_id'
)

if 'target' in full_features:
    target = full_features['target']
    del full_features['target']
classes = sorted(target.unique())

# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes : ', classes)
if 'object_id' in full_features:
    oof_df = full_features[['object_id']]
    del full_features['object_id'], full_features['distmod'], full_features['hostgal_specz']
    
    
train_mean = full_features.mean(axis=0)
full_features.fillna(train_mean, inplace=True)
full_features.head()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
clf_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'None',
    'learning_rate': 0.01,
    'subsample': .9,
    'colsample_bytree': .75,
    'reg_alpha': 1.0e-2,
    'reg_lambda': 1.0e-2,
    'min_split_gain': 0.01,
#     'min_child_weight': 10,
    'min_child_samples': 20,
#     'n_estimators': 2000,
#     'silent': -1,
#     'verbose': -1,
    'max_depth': 3,
    'importance_type': 'gain',
    'n_jobs': -1
}
boosters = []
importances = pd.DataFrame()
oof_preds = np.zeros((full_features.shape[0], target.unique().shape[0]))

warnings.simplefilter('ignore', FutureWarning)
for fold_id, (train_idx, validation_idx) in enumerate(folds.split(full_features, target)):
    print('processing fold {0}'.format(fold_id))
    X_train, y_train = full_features.iloc[train_idx], target.iloc[train_idx]
    X_valid, y_valid = full_features.iloc[validation_idx], target.iloc[validation_idx]

    train_dataset = lgb.Dataset(X_train, label_to_code(y_train))
    valid_dataset = lgb.Dataset(X_valid, label_to_code(y_valid))
    
    booster = lgb.train(clf_params.copy(), train_dataset, 
                        num_boost_round=2000,
                        fobj=wloss_objective, 
                        feval=wloss_metric,
                        valid_sets=[train_dataset, valid_dataset],
                        verbose_eval=100,
                        early_stopping_rounds=100
                       )
    oof_preds[validation_idx, :] = booster.predict(X_valid)

    imp_df = pd.DataFrame()
    imp_df['feature'] = full_features.columns
    imp_df['gain'] = booster.feature_importance('gain')
    imp_df['fold'] = fold_id
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    boosters.append(booster)
loss = multi_weighted_logloss(y_true=target, y_preds=softmax(oof_preds))
_, loss2, _ = wloss_metric(oof_preds, lgb.Dataset(full_features, label_to_code(target)))
print(f'OG wloss : {loss:.5f}, Re-implemented wloss: {loss2:.5f} ')  # 0.93526
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
plt.figure(figsize=(14, 14), facecolor='w')
sns.barplot(x='gain', y='feature', 
            data=importances.sort_values('mean_gain', ascending=False).iloc[:5 * 40])
plt.tight_layout()
plt.show()
meta_test = pd.read_csv('../input/test_set_metadata.csv')

import time

start = time.time()
chunks = 5000000
for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):
    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    # Group by object id
    agg_test = df.groupby('object_id').agg(aggs)
    agg_test.columns = new_columns
#     agg_test['mjd_diff'] = agg_test['mjd_max'] - agg_test['mjd_min']
    detected_groups = df[df['detected'] == 1].groupby('object_id')
    agg_test['det_mjd_diff'] = detected_groups['mjd'].transform('max') \
        - detected_groups['mjd'].transform('min')
    agg_test['flux_diff'] = agg_test['flux_max'] - agg_test['flux_min']
    agg_test['flux_dif2'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_mean']
    agg_test['flux_w_mean'] = agg_test['flux_by_flux_ratio_sq_sum'] / agg_test['flux_ratio_sq_sum']
    agg_test['flux_dif3'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_w_mean']

#     del agg_test['mjd_max'], agg_test['mjd_min']
    
    full_test = agg_test.reset_index().merge(
        right=meta_test,
        how='left',
        on='object_id'
    )
    full_test = full_test.fillna(train_mean)
    
    # Make predictions
    preds = None
    for booster in boosters:
        if preds is None:
            preds = softmax(booster.predict(full_test[full_features.columns])) / folds.n_splits
        else:
            preds += softmax(booster.predict(full_test[full_features.columns])) / folds.n_splits
    
   # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])
    
    # Store predictions
    preds_df = pd.DataFrame(preds, columns=['class_' + str(s) for s in classes])
    preds_df['object_id'] = full_test['object_id']
    preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99) 
    
    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, mode='a', index=False, float_format='%.6f')
    else: 
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False, float_format='%.6f')
        
    del agg_test, full_test, preds_df, preds
    gc.collect()
    
    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
z = pd.read_csv('predictions.csv')

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

z.to_csv('single_predictions.csv', index=True)