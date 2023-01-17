# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data['target'] = np.where(data['diagnosis'] == 'M', 1, 0)
ax = sns.countplot(data['target'],label="Count")       # M = 212, B = 357
B, M = data['target'].value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
data.drop('diagnosis', axis=1, inplace=True)
def cohen_effect_size(df, target):
    group1 = df[df[target]==0].drop(target, axis=1)
    group2 = df[df[target]==1].drop(target, axis=1)
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1 = group1.shape[0]
    n2 = group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d
df_ces = cohen_effect_size(data, 'target')
df_ces.head()
df_ces_s = df_ces.reindex(df_ces.abs().sort_values(ascending=False).index)
df_ces_s.head()
fig, ax = plt.subplots(figsize=(6, 10))
df_ces_s[df_ces_s.abs() > 0.4][::-1].plot.barh(ax=ax);
from sklearn.utils import shuffle

def p_value_effect(df, target, nr_iters=1000):
    actual = cohen_effect_size(df, target)
    results = np.zeros(actual.shape[0])
    df_shuffled = shuffle(df.drop(target, axis=1))
    for i in range(nr_iters):
        df_shuffled = shuffle(df_shuffled)
        df_shuffled[target] = df[target].values
        results = results + (cohen_effect_size(df_shuffled, target).abs() >= actual.abs())
    p_values = results/nr_iters
    return pd.DataFrame({'cohen_effect_size':actual, 'p_value':p_values}, index=actual.index)
df_ces_p = p_value_effect(data, 'target')
df_ces_p.head()
df_ces_p_s = df_ces_p.reindex(df_ces_p.cohen_effect_size.abs().sort_values(ascending=False).index)
fig, ax = plt.subplots(figsize=(6, 10))
df_ces_p_s[(df_ces_p_s.cohen_effect_size.abs() > 0.4) & (df_ces_p_s.p_value <= 0.05)][::-1].cohen_effect_size.plot.barh(ax=ax);
plt.title('Cohen effect size of the cancer features');
# What features have small effect size and what is their p value
df_ces_p_s[(df_ces_p_s.cohen_effect_size.abs() <= 0.4)]
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
X = data.drop('target', axis=1)
y = data['target']
X.shape
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1301, test_size=0.33)
# parameters for LightGBMClassifier
params = {
    'objective' :'binary',
    'learning_rate' : 0.33,
    'num_leaves' : 76,
    'feature_fraction': 0.64, 
    'bagging_fraction': 0.8, 
    'bagging_freq':1,
    'boosting_type' : 'gbdt',
    'metric': 'auc'
}
d_train = lgbm.Dataset(X_train, y_train)
d_valid = lgbm.Dataset(X_valid, y_valid)
    
    # training with early stop
bst = lgbm.train(params, d_train, 5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=10)
y_pred = bst.predict(X_valid, num_iteration=bst.best_iteration)
from sklearn.metrics import roc_curve, auc, roc_auc_score
roc_auc_score(y_valid, y_pred)
cv_result_lgb = lgbm.cv(params, d_train, 5000, verbose_eval=50, early_stopping_rounds=10, stratified=True, show_stdv=True, nfold=5)
len(cv_result_lgb['auc-mean'])
plt.errorbar(range(len(cv_result_lgb['auc-mean'])), cv_result_lgb['auc-mean'], yerr=cv_result_lgb['auc-stdv']);
plt.title('AUC + standard dev of each fold');
sig_features = df_ces_p_s[(df_ces_p_s.cohen_effect_size.abs() > 0.4) & (df_ces_p_s.p_value <= 0.05)].index.values
X = data[sig_features]
y = data['target']
X.shape
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1301, test_size=0.33)
d_train = lgbm.Dataset(X_train, y_train)
d_valid = lgbm.Dataset(X_valid, y_valid)
cv_result_lgb = lgbm.cv(params, d_train, 5000, verbose_eval=50, early_stopping_rounds=10, stratified=True, show_stdv=True, nfold=5)
plt.errorbar(range(len(cv_result_lgb['auc-mean'])), cv_result_lgb['auc-mean'], yerr=cv_result_lgb['auc-stdv']);
plt.title('AUC + standard dev of each fold');
bst = lgbm.train(params, d_train, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=10, num_boost_round=len(cv_result_lgb['auc-mean']))
y_pred = bst.predict(X_valid, num_iteration=bst.best_iteration)
roc_auc_score(y_valid, y_pred)
