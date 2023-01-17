%matplotlib inline



from functools import partial

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



from IPython.display import display_html



from sklearn.manifold import TSNE



from bayes_opt import BayesianOptimization



from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

import lightgbm as lgb

import warnings

warnings.simplefilter('ignore', category=FutureWarning)



pd.set_option('display.max_columns', 100)

sns.set_style('whitegrid')
PATH = Path('../input')

[f.name for f in PATH.iterdir()]
train_df = pd.read_csv(PATH/'flight_delays_train.csv')

print(train_df.shape)

train_df.head()
test_df = pd.read_csv(PATH/'flight_delays_test.csv')

print(test_df.shape)

test_df.head()
train_df['dep_delayed_15min'].value_counts(normalize=True)
def plot_by_target(df, col, plot=sns.countplot, hue='dep_delayed_15min', sharey=False):

    g = sns.FacetGrid(train_df, col=hue, height=4.5, aspect=1, sharey=sharey)

    g.map(plot, col)

    g.set_xticklabels(rotation=90);
plot_by_target(train_df, 'Month')
plot_by_target(train_df, col='DayofMonth')
plot_by_target(train_df, 'DayOfWeek')
plot_by_target(train_df, col='DepTime', plot=plt.hist)
plot_by_target(train_df, col='DepTime', plot=partial(sns.boxplot, showfliers=False))
plot_by_target(train_df, col='UniqueCarrier')
plot_by_target(train_df, col='Distance', plot=plt.hist)
plot_by_target(train_df, col='Distance', plot=partial(sns.boxplot, showfliers=False))
# Extract the labels

train_y = train_df.pop('dep_delayed_15min')

train_y = train_y.map({'N': 0, 'Y': 1})



# Concatenate for preprocessing

train_split = train_df.shape[0]

full_df = pd.concat((train_df, test_df))



# Hour and minute

full_df['hour'] = full_df['DepTime'] // 100

full_df.loc[full_df['hour'] == 24, 'hour'] = 0

full_df.loc[full_df['hour'] == 25, 'hour'] = 1

full_df['minute'] = full_df['DepTime'] % 100



# Season

full_df['summer'] = (full_df['Month'].isin(['c-6', 'c-7', 'c-8'])).astype(np.int32)

full_df['autumn'] = (full_df['Month'].isin(['c-9', 'c-10', 'c-11'])).astype(np.int32)

full_df['winter'] = (full_df['Month'].isin(['c-12', 'c-1', 'c-2'])).astype(np.int32)

full_df['spring'] = (full_df['Month'].isin(['c-3', 'c-4', 'c-5'])).astype(np.int32)



# Daytime

full_df['daytime'] = pd.cut(full_df['hour'], bins=[0, 6, 12, 18, 23], include_lowest=True)
# String to numerical

for col in ['Month', 'DayofMonth', 'DayOfWeek']:

    full_df[col] = full_df[col].apply(lambda x: x.split('-')[1]).astype(np.int32) - 1



# Label Encoding

for col in ['Origin', 'Dest', 'UniqueCarrier', 'daytime']:

    full_df[col] = pd.factorize(full_df[col])[0]



# Categorical columns

cat_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'Origin', 'Dest', 'UniqueCarrier', 'hour', 'summer', 'autumn', 'winter', 'spring', 'daytime']



# Converting categorical columns to type 'category' as required by LGBM

for c in cat_cols:

    full_df[c] = full_df[c].astype('category')



# Split into train and test

train_df, test_df = full_df.iloc[:train_split], full_df.iloc[train_split:]

train_df.shape, train_y.shape, test_df.shape
def cross_val_scheme(model, train_df, train_y, cv):

    cv_scores = cross_val_score(model, train_df, train_y, scoring='roc_auc', cv=cv, n_jobs=-1)

    print(f'CV Scores: {cv_scores}')

    print(f'CV mean: {cv_scores.mean()} \t CV Std: {cv_scores.std()}')

    model.fit(train_df, train_y)

    feat_imp = pd.DataFrame({'col': full_df.columns.values, 'imp': model.feature_importances_}).sort_values(by='imp', ascending=False)

    return cv_scores, feat_imp
skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

clf = lgb.LGBMClassifier(random_state=7)

cv_scores, feat_imp = cross_val_scheme(clf, train_df, train_y, skf)

plt.figure(figsize=(8, 10))

sns.barplot(x='imp', y='col', data=feat_imp[1:], orient='h');
bay_tr_ix, bay_val_ix = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=7).split(train_df, train_y))[0]
def LGB_bayesian(num_leaves, min_data_in_leaf, learning_rate, min_sum_hessian_in_leaf, feature_fraction, lambda_l1,

                 lambda_l2, min_gain_to_split, max_depth):

    

    num_leaves = int(np.round(num_leaves))

    min_data_in_leaf = int(np.round(min_data_in_leaf))

    max_depth = int(np.round(max_depth))

    

    params = {

        'feature_fraction': feature_fraction,

        'lambda_l1': lambda_l1,

        'lambda_l2': lambda_l2,

        'learning_rate': learning_rate,

        'max_depth': max_depth,

        'min_data_in_leaf': min_data_in_leaf,

        'min_gain_to_split': min_gain_to_split,

        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,

        'num_leaves': num_leaves,

        'max_bin': 255,

        'bagging_fraction': 0.8,

        'bagging_freq': 6,

        'save_binary': True,

        'seed': 7,

        'feature_fraction_seed': 7,

        'bagging_seed': 7,

        'drop_seed': 7,

        'data_random_seed': 7,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': True,

        'n_jobs': -1

    }

    

    lgb_train = lgb.Dataset(train_df.loc[bay_tr_ix], train_y.loc[bay_tr_ix], free_raw_data=False)

    lgb_valid = lgb.Dataset(train_df.loc[bay_val_ix], train_y.loc[bay_val_ix], free_raw_data=False)

    

    num_rounds=5000

    clf = lgb.train(params, lgb_train, num_rounds, valid_sets=[lgb_train, lgb_valid], verbose_eval=250, early_stopping_rounds=50)

    val_preds = clf.predict(train_df.loc[bay_val_ix], num_iterations=clf.best_iteration)

    

    score = roc_auc_score(train_y.loc[bay_val_ix], val_preds)

    return score





bounds_lgb = {

    'feature_fraction': (0.5, 1),

    'lambda_l1': (0., 10.),

    'lambda_l2': (0., 10.),

    'learning_rate': (0.01, 0.1),

    'max_depth': (2, 8),

    'min_data_in_leaf': (5, 30),

    'min_gain_to_split': (0, 1),

    'min_sum_hessian_in_leaf': (0.01, 1),

    'num_leaves': (10, 35)

}



LGB_BO = BayesianOptimization(LGB_bayesian, bounds_lgb, random_state=7)



with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=10, n_iter=10, acq='ucb')
LGB_BO.max['target'], LGB_BO.max['params']
def test_kfold(params, train_df, train_y, test_df, cv):

    test_preds = 0.

    valid_preds = np.zeros(train_y.shape)

    

    for fold, (train_ix, valid_ix) in enumerate(cv.split(train_df, train_y)):

        print(f"\nFOLD: {fold+1} {'='*50}")

        X_train, X_valid = train_df.iloc[train_ix], train_df.iloc[valid_ix]

        y_train, y_valid = train_y.iloc[train_ix], train_y.iloc[valid_ix]

        

        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

        lgb_valid = lgb.Dataset(X_valid, y_valid, free_raw_data=False)

        

        clf = lgb.train(params, lgb_train, 5000, valid_sets=[lgb_train, lgb_valid], verbose_eval=250, early_stopping_rounds=50)

        valid_preds[valid_ix] = clf.predict(train_df.iloc[valid_ix], num_iterations=clf.best_iteration)

        test_preds += clf.predict(test_df, num_iterations=clf.best_iteration)

    

    print(f'Valid CV: {roc_auc_score(train_y, valid_preds)}')

    test_preds /= cv.n_splits

    

    return test_preds
params = {

    'feature_fraction': LGB_BO.max['params']['feature_fraction'],

    'lambda_l1': LGB_BO.max['params']['lambda_l1'],

    'lambda_l2': LGB_BO.max['params']['lambda_l2'],

    'learning_rate': LGB_BO.max['params']['learning_rate'],

    'max_depth': int(np.round(LGB_BO.max['params']['max_depth'])),

    'min_data_in_leaf': int(np.round(LGB_BO.max['params']['min_data_in_leaf'])),

    'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],

    'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],

    'num_leaves': int(np.round(LGB_BO.max['params']['num_leaves'])),

    'max_bin': 255,

    'bagging_fraction': 0.8,

    'bagging_freq': 3,

    'save_binary': True,

    'seed': 7,

    'feature_fraction_seed': 7,

    'bagging_seed': 7,

    'drop_seed': 7,

    'data_random_seed': 7,

    'objective': 'binary',

    'boosting_type': 'gbdt',

    'verbose': 1,

    'metric': 'auc',

    'is_unbalance': True,

    'boost_from_average': True,

    'n_jobs': -1

}



# 5-Fold testing

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')    

    test_preds = test_kfold(params, train_df, train_y, test_df, StratifiedKFold(n_splits=5, random_state=7, shuffle=True))
final_df = pd.DataFrame({'id': range(test_preds.shape[0]), 'dep_delayed_15min': test_preds})

final_df.to_csv('lightgbm_5fold_sub_v2.csv', header=True, index=False)

pd.read_csv('lightgbm_5fold_sub_v2.csv').head()