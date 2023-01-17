# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

from lightgbm import LGBMClassifier
import gc
import pandas_profiling
import matplotlib.pyplot as plt
def cohen_effect_size(X, y):
    """Calculates the Cohen effect size of each feature.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
        Notes
        -----
        Based on https://github.com/AllenDowney/CompStats/blob/master/effect_size.ipynb
    """
    group1, group2 = X[y==0], X[y==1]
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = group1.shape[0], group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d
LoanID = 'SK_ID_CURR'
data   = pd.read_csv('../input/application_train.csv').set_index(LoanID)
test   = pd.read_csv('../input/application_test.csv').set_index(LoanID)
prev   = pd.read_csv('../input/previous_application.csv')
buro   = pd.read_csv('../input/bureau.csv')
burobl = pd.read_csv('../input/bureau_balance.csv')
credit = pd.read_csv('../input/credit_card_balance.csv')
def PivotGroupBy(df, groupby_id, target_id, feature_name='', cutoff=0.05):
    cnt_name = 'cnt_{0}'.format(target_id)
    tmp = df.groupby([groupby_id])[target_id].value_counts(normalize=True)
    tmp = tmp.loc[tmp >= cutoff].rename(cnt_name).reset_index()
    tmp = tmp.pivot(index=groupby_id, columns=target_id, values=cnt_name)
    tmp.rename(columns={f:'{0}_r={1}'.format(feature_name, f) for f in tmp.columns}, inplace=True)
    return tmp

def CatMeanEnc(df, index_name, groupby_ids):
###################################
# PLEASE DON'T DO THIS AT HOME LOL
# Averaging factorized categorical features defeats my own reasoning
################################### 
    cat_features = [f_ for f_ in df.columns if df[f_].dtype == 'object']
    
    df_pivots = [PivotGroupBy(df, index_name, f_, feature_name=f_) for f_ in cat_features]
    df_pivots = pd.concat(df_pivots, axis=1)
#    for f_ in cat_features:
#        df[f_], _ = pd.factorize(df[f_])

    df_ret = df[[f for f in df.columns if f not in cat_features]].groupby(LoanID).mean().join(df_pivots, how='left')
    print(df_ret.describe())
    df_ret['cnt_{:}'.format(index_name)] = df[[groupby_ids, index_name]].groupby(groupby_ids).count()[index_name]
    del df_ret[index_name]
    return df_ret

def JoinMeanEnc(main_df, join_dfs=[]):
    for df in join_dfs:
        print(main_df.shape, df.shape)
        f_join = [f_ for f_ in df.columns if f_ not in main_df.columns]
        main_df = main_df.join(df[f_join], how='left')
    return main_df
# Attach bureau_balance to Bureau
tmp = PivotGroupBy(burobl, 'SK_ID_BUREAU', 'STATUS', feature_name='bureau_balance')
tmp['LONGEST_MONTHS'] = burobl.groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].size()
buro = buro.join(tmp, how='left', on='SK_ID_BUREAU')
print(buro.head())
del burobl, tmp

# factorize         
categorical_feats = [f for f in data.columns if data[f].dtype == 'object']
for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])
    
print(data.shape, test.shape)

y = data['TARGET']
del data['TARGET']

avg_dfs = [CatMeanEnc(prev,   index_name='SK_ID_PREV', groupby_ids=LoanID), 
           CatMeanEnc(buro,   index_name='SK_ID_BUREAU', groupby_ids=LoanID), 
           CatMeanEnc(credit, index_name='SK_ID_PREV', groupby_ids=LoanID)]
data = JoinMeanEnc(data, join_dfs=avg_dfs)
test = JoinMeanEnc(test, join_dfs=avg_dfs)

excluded_feats = [] #['SK_ID_CURR']
features = [f_ for f_ in data.columns if f_ not in excluded_feats]
print('Number of features %d' % len(features))
effect_sizes = cohen_effect_size(data[features], y)
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(30).index)[::-1].plot.barh(figsize=(6, 10));
plt.title('Features with the 30 largest effect sizes');
significant_features = [f for f in features if np.abs(effect_sizes.loc[f]) > 0.1]
print('Significant features %d: %s' % (len(significant_features), significant_features))
profile = pandas_profiling.ProfileReport(data[significant_features])
profile
rejected_variables = profile.get_rejected_variables(threshold=0.9)
rejected_variables
selected_features = list(set(significant_features) - set(rejected_variables))
selected_features
X = data[selected_features].copy()
# X['EXT_SOURCE_1'].fillna((X['EXT_SOURCE_1'].mean()), inplace=True)
# X['EXT_SOURCE_2'].fillna((X['EXT_SOURCE_2'].median()), inplace=True)
# X['EXT_SOURCE_3'].fillna((X['EXT_SOURCE_3'].median()), inplace=True)
# X['CNT_DRAWINGS_ATM_CURRENT'].fillna((X['CNT_DRAWINGS_ATM_CURRENT'].median()), inplace=True)
# X['DAYS_CREDIT'].fillna((X['DAYS_CREDIT'].mean()), inplace=True)
# X['AMT_BALANCE'].fillna(0, inplace=True)
# Fill the remaining Nan's with zero
X.fillna(0, inplace=True)
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_sample(X, y)
params_LGBM = {
    'n_estimators'     : 4000,
    'learning_rate'    : 0.03,
    'num_leaves'       : 70,
    'colsample_bytree' : 0.8,
    'subsample'        : 0.9,
    'max_depth'        : 7,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 0.1,
    'min_split_gain'   : 0.01,
    'min_child_weight' : 2,
    'silent'           : True,
    }
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1301)
oof_preds = np.zeros(X_resampled.shape[0])
sub_preds = np.zeros(test.shape[0])
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_resampled, y_resampled)):
    trn_x, trn_y = X_resampled[trn_idx], y_resampled[trn_idx]
    val_x, val_y = X_resampled[val_idx], y_resampled[val_idx]
    
    clf = LGBMClassifier(**params_LGBM)
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=150
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[selected_features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()

score = roc_auc_score(y_resampled, oof_preds)
print('Full AUC score %.6f' % score)
test['TARGET'] = sub_preds
test[['TARGET']].to_csv('subm_lgbm_auc{:.8f}.csv'.format(score), index=True, float_format='%.8f')
