import numpy as np 

import pandas as pd

import lightgbm as lgb

from scipy.stats import norm

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import roc_auc_score
train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)



target = train.target.values

train.drop('target', axis=1, inplace=True)

train.shape, target.shape, test.shape, 
pos_idx = (target == 1)

neg_idx = (target == 0)

stats = []

for col in train.columns:

    stats.append([

        train.loc[pos_idx, col].mean(),

        train.loc[pos_idx, col].std(),

        train.loc[neg_idx, col].mean(),

        train.loc[neg_idx, col].std()

    ])

    

stats_df = pd.DataFrame(stats, columns=['pos_mean', 'pos_sd', 'neg_mean', 'neg_sd'])

stats_df.head()
npos = pos_idx.sum()

nneg = neg_idx.sum()



sim_feats = []

for pos_mean, pos_sd, neg_mean, neg_sd in stats:

    pos_feat = np.random.normal(loc=pos_mean, scale=pos_sd, size=npos)

    neg_feat = np.random.normal(loc=neg_mean, scale=neg_sd, size=nneg)

    sim_feats.append(np.hstack([pos_feat, neg_feat]))

    

sim_feats = np.column_stack(sim_feats)

sim_target = np.hstack([np.ones(npos), np.zeros(nneg)])
sim_feats.shape, sim_target.shape
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.335,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.041,

    'learning_rate': 0.0083,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': -1

}
trn_data = lgb.Dataset(sim_feats, sim_target)

cv = lgb.cv(param, trn_data, 100000, shuffle=True, early_stopping_rounds=600, verbose_eval=600)

print(cv['auc-mean'][-1], len(cv['auc-mean']))
plt.figure(figsize=(20, 10))

# var_0

plt.subplot(2, 2, 1)

sns.distplot(train.loc[pos_idx, 'var_0'], hist=False, label='pos', color='blue')

sns.distplot(train.loc[neg_idx, 'var_0'], hist=False, label='neg', color='orange')

plt.vlines(x=[stats_df.loc[0, 'pos_mean'], stats_df.loc[0, 'neg_mean']], ymin=0, ymax=0.15, colors=['blue', 'orange'])

plt.xlabel('var_0')

plt.title('Real data')

plt.legend()

plt.subplot(2, 2, 2)

sns.distplot(sim_feats[pos_idx, 0], hist=False, label='pos', color='blue')

sns.distplot(sim_feats[neg_idx, 0], hist=False, label='neg', color='orange')

plt.vlines(x=[stats_df.loc[0, 'pos_mean'], stats_df.loc[0, 'neg_mean']], ymin=0, ymax=0.15, colors=['blue', 'orange'])

plt.title('Simulated data')

plt.legend()

plt.xlabel('var_0')



# var_1

plt.subplot(2, 2, 3)

sns.distplot(train.loc[pos_idx, 'var_1'], hist=False, label='pos', color='blue')

sns.distplot(train.loc[neg_idx, 'var_1'], hist=False, label='neg', color='orange')

plt.vlines(x=[stats_df.loc[1, 'pos_mean'], stats_df.loc[1, 'neg_mean']], ymin=0, ymax=0.15, colors=['blue', 'orange'])

plt.xlabel('var_1')

plt.legend()

plt.subplot(2, 2, 4)

sns.distplot(sim_feats[pos_idx, 1], hist=False, label='pos', color='blue')

sns.distplot(sim_feats[neg_idx, 1], hist=False, label='neg', color='orange')

plt.vlines(x=[stats_df.loc[1, 'pos_mean'], stats_df.loc[1, 'neg_mean']], ymin=0, ymax=0.15, colors=['blue', 'orange'])

plt.legend()

plt.xlabel('var_1')
zval1 = (train.values - stats_df.neg_mean.values) / stats_df.neg_sd.values

zval1.shape
pval1 = (1 - norm.cdf(np.abs(zval1))) * 2
pval1
prob1 = pval1.prod(axis=1)
roc_auc_score(target, 1/prob1)
zval2 = (train.values - stats_df.pos_mean.values) / stats_df.pos_sd.values

pval2 = (1 - norm.cdf(np.abs(zval2))) * 2
prob2 = pval2.prod(axis=1)
roc_auc_score(target, prob2 / prob1)
te_zval1 = (test.values - stats_df.neg_mean.values) / stats_df.neg_sd.values

te_pval1 = (1 - norm.cdf(np.abs(te_zval1))) * 2

te_prob1 = te_pval1.prod(axis=1)
te_zval2 = (test.values - stats_df.pos_mean.values) / stats_df.pos_sd.values

te_pval2 = (1 - norm.cdf(np.abs(te_zval2))) * 2

te_prob2 = te_pval2.prod(axis=1)
pred = te_prob2 / te_prob1
pd.DataFrame({

    'ID_code': test.index,

    'target': pred

}).to_csv('sub.csv', index=False)