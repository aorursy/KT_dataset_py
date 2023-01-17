# The number of entries to read in. Use it to have fast turn-around. The values are separate for train and test sets
max_events_trn=None
max_events_tst=None
# Number on CV folds
n_cv=3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=Warning)

from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
print(os.listdir("../input"))
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
df_trn = pd.read_csv('../input/train.csv', nrows=max_events_trn)
df_trn = reduce_mem_usage(df_trn)

df_tst = pd.read_csv('../input/test.csv',  nrows=max_events_tst)
df_tst = reduce_mem_usage(df_tst)
df_trn.head()
df_trn.info(memory_usage='deep', verbose=False)
df_tst.info(memory_usage='deep', verbose=False)
df_trn.isnull().sum().sum()
y = df_trn['winPlacePerc']
df_trn.drop('winPlacePerc', axis=1, inplace=True)
# we will NOT use 
features_not2use = ['Id', 'groupId', 'matchId']
for df in [df_trn, df_tst]:
    df.drop(features_not2use, axis=1, inplace=True)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone, ClassifierMixin, RegressorMixin
import lightgbm as lgb


def learning_rate_decay_power(current_iter):
    '''
    The function defines learning rate deay for LGBM
    '''
    base_learning_rate = 0.10
    min_lr = 5e-2
    lr = base_learning_rate  * np.power(.996, current_iter)
    return lr if lr > min_lr else min_lr


def train_single_model(clf_, X_, y_, random_state_=314, opt_parameters_={}, fit_params_={}):
    '''
    A wrapper to train a model with particular parameters
    '''
    c = clone(clf_)
    c.set_params(**opt_parameters_)
    c.set_params(random_state=random_state_)
    return c.fit(X_, y_, **fit_params_)

def train_model_in_CV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            seed=31416, n=5,
                            opt_parameters_={}, fit_params_={},
                            verbose=True):
    # the list of classifiers for voting ensable
    clfs = []
    # performance 
    perf_eval = {'score_i_oof': 0,
                 'score_i_ave': 0,
                 'score_i_std': 0,
                 'score_i': []
                }
    # full-sample oof prediction
    y_full_oof = pd.Series(np.zeros(shape=(y.shape[0],)), 
                          index=y.index)
    
    if 'sample_weight' in metric_args:
        sample_weight=metric_args['sample_weight']
        
    doSqrt=False
    if 'sqrt' in metric_args:
        doSqrt=True
        del metric_args['sqrt']

    cv = KFold(n, shuffle=True, random_state=seed) #Stratified
    # The out-of-fold (oof) prediction for the k-1 sample in the outer CV loop
    y_oof = pd.Series(np.zeros(shape=(X.shape[0],)), 
                      index=X.index)
    scores = []
    clfs = []

    for n_fold, (trn_idx, val_idx) in enumerate(cv.split(X, (y!=0).astype(np.int8))):
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if fit_params_:
            # use _stp data for early stopping
            fit_params_["eval_set"] = [(X_trn,y_trn), (X_val,y_val)]
            fit_params_['verbose'] = verbose

        clf = train_single_model(model, X_trn, y_trn, 314+n_fold, opt_parameters_, fit_params_)

        clfs.append(('{}{}'.format(model_name,n_fold), clf))
        # evaluate performance
        if isinstance(clf, RegressorMixin):
            y_oof.iloc[val_idx] = clf.predict(X_val)
        elif isinstance(clf, ClassifierMixin):
            y_oof.iloc[val_idx] = clf.predict_proba(X_val)[:,1]
        else:
            raise TypeError('Provided model does not inherit neither from a regressor nor from classifier')
        if 'sample_weight' in metric_args:
            metric_args['sample_weight'] = y_val.map(sample_weight)
        scores.append(metric(y_val, y_oof.iloc[val_idx], **metric_args))
        #cleanup
        del X_trn, y_trn, X_val, y_val

    # Store performance info for this CV
    if 'sample_weight' in metric_args:
        metric_args['sample_weight'] = y_oof.map(sample_weight)
    perf_eval['score_i_oof'] = metric(y, y_oof, **metric_args)
    perf_eval['score_i'] = scores
    
    if doSqrt:
        for k in perf_eval.keys():
            if 'score' in k:
                perf_eval[k] = np.sqrt(perf_eval[k])
        scores = np.sqrt(scores)
            
    perf_eval['score_i_ave'] = np.mean(scores)
    perf_eval['score_i_std'] = np.std(scores)

    return clfs, perf_eval, y_oof

def print_perf_clf(name, perf_eval):
    print('Performance of the model:')    
    print('Mean(Val) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                      perf_eval['score_i_ave'],
                                                                      perf_eval['score_i_std']
                                                                     ))
    print('Min/max scores on folds: {:.4f} / {:.4f}'.format(np.min(perf_eval['score_i']),
                                                            np.max(perf_eval['score_i'])))
    print('OOF score inner {} Classifier: {:.4f}'.format(name, perf_eval['score_i_oof']))
    print('Scores in individual folds: {}'.format(perf_eval['score_i']))
mdl_inputs = {
        # This will be with MAE loss
            'lgbm1_reg': (lgb.LGBMRegressor(max_depth=-1, min_child_samples=400, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000, learning_rate=0.05),
                 {'objective': 'mae', 'colsample_bytree': 0.75, 'min_child_weight': 10.0, 'num_leaves': 30, 'reg_alpha': 1, 'subsample': 0.75}, 
                 {"early_stopping_rounds":100, 
                  "eval_metric" : 'mae',
                  'eval_names': ['train', 'early_stop'],
                  'verbose': False, 
                  'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_decay_power)],
                  'categorical_feature': 'auto'},
                 y
                ),
#         'lgbm45_reg': (lgb.LGBMRegressor(max_depth=-1, min_child_samples=400, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000, learning_rate=0.05),
#                  {'objective': 'mae', 'colsample_bytree': 0.75, 'min_child_weight': 10.0, 'num_leaves': 45, 'reg_alpha': 1, 'subsample': 0.75}, 
#                  {"early_stopping_rounds":100, 
#                   "eval_metric" : 'mae',
#                   'eval_names': ['train', 'early_stop'],
#                   'verbose': False, 
#                   'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_decay_power)],
#                   'categorical_feature': 'auto'},
#                  y
#                 ),
#         'lgbm60_reg': (lgb.LGBMRegressor(max_depth=-1, min_child_samples=400, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000, learning_rate=0.05),
#                  {'objective': 'mae', 'colsample_bytree': 0.75, 'min_child_weight': 10.0, 'num_leaves': 60, 'reg_alpha': 1, 'subsample': 0.75}, 
#                  {"early_stopping_rounds":100, 
#                   "eval_metric" : 'mae',
#                   'eval_names': ['train', 'early_stop'],
#                   'verbose': False, 
#                   'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_decay_power)],
#                   'categorical_feature': 'auto'},
#                  y
#                 ),
#         'lgbm90_reg': (lgb.LGBMRegressor(max_depth=-1, min_child_samples=400, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000, learning_rate=0.05),
#                  {'objective': 'mae', 'colsample_bytree': 0.75, 'min_child_weight': 10.0, 'num_leaves': 90, 'reg_alpha': 1, 'subsample': 0.75}, 
#                  {"early_stopping_rounds":100, 
#                   "eval_metric" : 'mae',
#                   'eval_names': ['train', 'early_stop'],
#                   'verbose': False, 
#                   'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_decay_power)],
#                   'categorical_feature': 'auto'},
#                  y
#                 ),
        # This will be with FAIR loss
#         'lgbm2_reg': (lgb.LGBMRegressor(max_depth=-1, min_child_samples=400, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000, learning_rate=0.05),
#                  {'objective': 'fair', 'colsample_bytree': 0.75, 'min_child_weight': 10.0, 'num_leaves': 30, 'reg_alpha': 1, 'subsample': 0.75}, 
#                  {"early_stopping_rounds":100, 
#                   "eval_metric" : 'mae',
#                   'eval_names': ['train', 'early_stop'],
#                   'verbose': False, 
#                   'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_decay_power)],
#                   'categorical_feature': 'auto'},
#                  y
#                 ),
       }
%%time
mdls = {}
results = {}
y_oofs = {}
for name, (mdl, mdl_pars, fit_pars, y_) in mdl_inputs.items():
    print('--------------- {} -----------'.format(name))
    mdl_, perf_eval_, y_oof_ = train_model_in_CV(mdl, df_trn, y_, mean_absolute_error, 
                                                          metric_args={},
                                                          model_name=name, 
                                                          opt_parameters_=mdl_pars,
                                                          fit_params_=fit_pars, 
                                                          n=n_cv,
                                                          verbose=False)
    results[name] = perf_eval_
    mdls[name] = mdl_
    y_oofs[name] = y_oof_
    print_perf_clf(name, perf_eval_)
k = list(y_oofs.keys())[0]
_ = y_oofs[k].plot('hist', bins=100, figsize=(15,6))
plt.xlabel('Predicted winPlacePerc OOF')
import matplotlib.pyplot as plt
import seaborn as sns

def display_importances(feature_importance_df_, n_feat=30, silent=False, dump_strs=[], 
                        fout_name=None, title='Features (avg over folds)'):
    '''
    Make a plot of most important features from a tree-based model

    Parameters
    ----------
    feature_importance_df_ : pd.DataFrame
        The input dataframe. 
        Must contain columns `'feature'` and `'importance'`.
        The dataframe will be first grouped by `'feature'` and the mean `'importance'` will be calculated.
        This allows to calculate and plot importance averaged over folds, 
        when the same features appear in the dataframe as many time as there are folds in CV.
    n_feats : int [default: 20]
        The maximum number of the top features to be plotted
    silent : bool [default: False]
        Dump additionsl information, in particular the mean importances for features 
        defined by `dump_strs` and the features with zero (<1e-3) importance
    dump_strs : list of strings [default: []]
        Features containing either of these srings will be printed to the screen
    fout_name : str or None [default: None]
        The name of the file to dump the figure. 
        If `None`, no file is created (to be used in notebooks)
    '''
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_feat].index  
    
    mean_imp = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    df_2_neglect = mean_imp[mean_imp['importance'] < 1e-3]
    
    if not silent:
        print('The list of features with 0 importance: ')
        print(df_2_neglect.index.values.tolist())

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        for feat_prefix in dump_strs:
            feat_names = [x for x in mean_imp.index if feat_prefix in x]
            print(mean_imp.loc[feat_names].sort_values(by='importance', ascending=False))
    del mean_imp, df_2_neglect
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title(title)
    plt.tight_layout()

    if fout_name is not None:
        plt.savefig(fout_name)

display_importances(pd.DataFrame({'feature': df_trn.columns,
                                  'importance': mdls['lgbm1_reg'][0][1].booster_.feature_importance('gain')}),
                    n_feat=20,
                    title='GAIN feature importance',
                    fout_name='feature_importance_gain.png'
                   )
%%time
y_subs= {}
for c in mdl_inputs:
    mdls_= mdls[c]
    y_sub = np.zeros(df_tst.shape[0])
    for mdl_ in mdls_:
        y_sub += np.clip(mdl_[1].predict(df_tst), 0, 1)
    y_sub /= n_cv
    
    y_subs[c] = y_sub
df_sub = pd.read_csv('../input/sample_submission.csv', nrows=max_events_tst)
for c in mdl_inputs:
    df_sub['winPlacePerc'] = y_subs[c]
    df_sub.to_csv('sub_{}.csv'.format(c), index=False)
    
    oof = pd.DataFrame(y_oofs[c].values)
    oof.columns = ['winPlacePerc']
    oof.clip(0, 1, inplace=True)
    oof.to_csv('oof_{}.csv'.format(c), index=False)
!ls
