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
import shap
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
    # print(df_ret.describe())
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
from sklearn.model_selection import train_test_split
X = data[significant_features].fillna(0)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=1301)
from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_jobs=-1)
# from sklearn.model_selection import GridSearchCV
# param_grid = { 'max_depth' : [12, 13, 14], 'n_estimators' : [400]}
# grid = GridSearchCV(clf, param_grid=param_grid, scoring = "roc_auc") 
# grid.fit(X_train, y_train)
# print("Grid-Search with AUC" ) 
# print("Best parameters:" , grid.best_params_ )
# print("Best cross-validation train score (AUC)): {:.3f}".format(grid.best_score_))
# y_hat = grid.predict_proba(X_valid)[:, 1]
# score = roc_auc_score(y_valid, y_hat)
# print("Valid set AUC: {:.3f}" .format(score))
# y_hat = grid.predict_proba(X)[:, 1]
# score = roc_auc_score(y, y_hat)
# print("Overall AUC: {:.3f}" .format(score))
clf = RandomForestClassifier(max_depth=10, n_estimators=300)
clf.fit(X, y)
y_hat = clf.predict_proba(X)[:, 1]
score = roc_auc_score(y, y_hat)
# load JS visualization code to notebook
shap.initjs() 
shap_values = shap.TreeExplainer(clf).shap_values(X)
shap_values[0]
shap.summary_plot(shap_values[0], X)
shap.dependence_plot("EXT_SOURCE_3", shap_values[0], X)
shap.dependence_plot("EXT_SOURCE_1", shap_values[0], X)
shap.dependence_plot("EXT_SOURCE_2", shap_values[0], X)
shap.dependence_plot("DAYS_CREDIT", shap_values[0], X)
shap.dependence_plot("DAYS_BIRTH", shap_values[0], X)
shap.dependence_plot("DAYS_CREDIT_UPDATE", shap_values[0], X)
shap.dependence_plot("NAME_INCOME_TYPE", shap_values[0], X)
shap.dependence_plot("DAYS_CREDIT_ENDDATE", shap_values[0], X)
shap.dependence_plot("CNT_DRAWINGS_ATM_CURRENT", shap_values[0], X)
shap.dependence_plot("CODE_GENDER", shap_values[0], X)
shap.dependence_plot("CNT_DRAWINGS_CURRENT", shap_values[0], X)
sub_pred = clf.predict_proba(test[significant_features].fillna(0))[:, 1]
test['TARGET'] = sub_pred
test[['TARGET']].to_csv('subm_rfc_auc{:.8f}.csv'.format(score), index=True, float_format='%.8f')
