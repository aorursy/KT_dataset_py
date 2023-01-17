import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import gc
import time
from datetime import datetime
import warnings
warnings.simplefilter(action = 'ignore')
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
import lightgbm as lgbm
def reduce_mem_usage(df_, max_reduce = True):
    start_mem = df_.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
    
    for c in df_.columns[df_.dtypes != 'object']:
        col_type = df_[c].dtype
        
        c_min = df_[c].min()
        c_max = df_[c].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_[c] = df_[c].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_[c] = df_[c].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_[c] = df_[c].astype(np.int32)
            else:
                df_[c] = df_[c].astype(np.int64)  
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and max_reduce:
                df_[c] = df_[c].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_[c] = df_[c].astype(np.float32)
            else:
                df_[c] = df_[c].astype(np.float64)

    end_mem = df_.memory_usage().sum() / 1024**2
    print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df_
def new_features_from_version(df_, feature_, target_):
    
    def version_to_int(value):
        format_mask = ['{:0>2}', '{:0>3}', '{:0>5}', '{:0>5}']
        new_value = [f.format(v) for f, v in zip(format_mask, str(value).split('.'))]
        return int(new_value[0] + new_value[1] + new_value[2] + new_value[3])
    
    def split_column(df_, feature_, sep):
        res = pd.DataFrame.from_records(list(df_[feature_].astype('object').apply(lambda x: tuple(str(x).split(sep))).values),
                                        index = df_.index)
        res.columns = [feature_ + '_' + str(c) for c in res.columns]
        return res

    new_features = pd.DataFrame(index = df_.index)
    
    # convert version into numerical representation
    new_features[feature_ + '_int'] = df_[feature_].apply(version_to_int).astype(int)
    new_features[feature_ + '_int'] -= new_features[feature_ + '_int'].min() # for reduce memory
    
    # frequencies encoding    
    new_features[feature_ + '_frq'] = df_[feature_].map(df_[feature_].value_counts())
    
    # split version
    split = split_column(df_, feature_, '.')
    split['target'] = target_
    
    # features from major, minor, build, revision if it has sense
    nun = split.nunique(dropna = False)
    if sum(nun < 2) > 0:
        to_drop = list(split.columns[nun < 2])
        split.drop(to_drop, axis = 1, inplace = True)

    # target encoding for risk zones
    to_drop = []
    for c in split.columns.drop('target'):
        if split[c].value_counts(dropna = False).iloc[1] < 100000:
            to_drop.append(c)
        else:
            new_features[c] = split[c].astype(int)
    split.drop(to_drop, axis = 1, inplace = True)
    
    for c in split.columns.drop('target'):
        te = split.groupby([c])['target'].transform(np.mean)
        split[c + '_good'] = -(te <= .4).astype(int)
        split[c + '_bad'] =  (te >= .6).astype(int)
        
    # risk zones feature 
    new_features[feature_ + '_risk'] = split.drop('target', axis = 1).sum(axis = 1)
    
    new_features = reduce_mem_usage(new_features)
    
    return new_features
def lgbm_fast_check(df_, target_, params_):
    
    params = params_.copy()
    params['metric'] = 'auc'

    train_data = lgbm.Dataset(data = df_, label = target_)
    clf = lgbm.train(params, 
                     train_set = train_data, valid_sets = [train_data], 
                     num_boost_round = 100, verbose_eval = 10, keep_training_booster = True)
        
    pred = clf.predict(df_)
    
    importances = pd.DataFrame(index = df_.columns)
    importances['cnt'] = pd.Series(clf.feature_importance(), index = df_.columns)
    importances['gain'] = pd.Series(clf.feature_importance(importance_type = 'gain'), index = df_.columns)
    importances.fillna(0, inplace = True)
    
    return importances, \
           [roc_auc_score(target_, pred), log_loss(target_, pred), accuracy_score(target_, (pred >= .5) * 1)]
def lgbm_cross_validation(df_, target_, params_,
                          num_boost_round = 20000, early_stopping_rounds = 200,
                          model_prefix = '',
                          num_folds = 3, rs = 0, verbose = 100):
    
    print(params_)
    
    clfs = []
    importances_cnt = pd.DataFrame(index = df_.columns)
    importances_gain = pd.DataFrame(index = df_.columns)
    folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = rs)
    
    valid_pred = np.zeros(df_.shape[0])
    
    # Cross-validation cycle
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(target_, target_)):
        print('--- Fold {} started at {}'.format(n_fold, time.ctime()))
        
        train_x, train_y = df_.iloc[train_idx], target_.iloc[train_idx]
        valid_x, valid_y = df_.iloc[valid_idx], target_.iloc[valid_idx]
    
        train_data = lgbm.Dataset(data = train_x, label = train_y)
        valid_data = lgbm.Dataset(data = valid_x, label = valid_y, reference = train_data)
        
        clf = lgbm.train(params_, 
                         train_set = train_data, valid_sets = [train_data, valid_data], 
                         num_boost_round = num_boost_round, early_stopping_rounds = early_stopping_rounds, 
                         verbose_eval = verbose, keep_training_booster = True)
        
        clfs.append(clf)
        if len(model_prefix) > 0:
            clf.save_model(model_prefix + str(n_fold) + '.txt')

        valid_pred[valid_idx] = clf.predict(valid_x)
    
        valid_score = valid_pred[valid_idx]
        tn, fp, fn, tp = confusion_matrix(valid_y, (valid_score >= .5) * 1).ravel()
        ras = roc_auc_score(valid_y, valid_score)
        acc = accuracy_score(valid_y, (valid_score >= .5) * 1)
        loss = log_loss(valid_y, valid_score)
        print('--- Final valid score for this fold ---')
        print('TN =', tn, 'FN =', fn, 'FP =', fp, 'TP =', tp)
        print('AUC = ', ras, 'Loss =', loss, 'Acc =', acc)
        print('-'*40)

        importances_cnt[n_fold] = pd.Series(clf.feature_importance(), index = df_.columns)
        importances_gain[n_fold] = pd.Series(clf.feature_importance(importance_type = 'gain'), 
                                             index = df_.columns)
        
        del train_x, train_y, valid_x, valid_y, train_data, valid_data, valid_score
        gc.collect()

    importances = pd.DataFrame(index = df_.columns)
    importances['cnt'] = importances_cnt.mean(axis = 1)
    importances['gain'] = importances_gain.mean(axis = 1)
    importances.fillna(0, inplace = True)
    
    return clfs, valid_pred, importances
parameters = {}
parameters['device'] = 'cpu'
parameters['objective'] = 'binary'
parameters['n_jobs'] = -1
parameters['boosting'] = 'gbdt'
parameters['two_round'] = True
parameters['learning_rate'] = .05           # default = 0.1
parameters['feature_fraction'] = .8         # default = 1.
parameters['bagging_freq'] = 1              # default = 0
parameters['bagging_fraction'] = .3         # default = 1.
parameters['max_depth'] = -1                # default = -1 
parameters['num_leaves'] = 20               # default = 31 
parameters['max_bin'] = 1024                # default = 255, bigger is only for CPU!
parameters['min_data_in_leaf'] = 100        # default = 20
parameters['lambda_l1'] = 100.              # default = 0
parameters['lambda_l2'] = 100.              # default = 0
parameters['random_seed'] = 0
protection_features = [
    'HasTpm',
    'ProductName',
    'AVProductStatesIdentifier', 
    'AVProductsInstalled', 
    'AVProductsEnabled',
    'IsProtected', 
    'SMode', 
    'SmartScreen', 
    'Firewall',
    'UacLuaenable',
    'Census_IsSecureBootEnabled'
]
version_num_features = ['AvSigVersion'] # only one for Kaggle kernel
train = pd.read_csv('../input/train.csv', 
                    usecols = ['MachineIdentifier', 'HasDetections'] + protection_features + version_num_features)
train.drop(5244810, inplace = True) #bad AvSigVersion value
train.set_index('MachineIdentifier', inplace = True)
train.head()
target_train = train['HasDetections']
train.drop('HasDetections', axis = 1, inplace = True)
target_train.value_counts()
test = pd.read_csv('../input/test.csv', 
                   usecols = ['MachineIdentifier'] + protection_features + version_num_features)
test.set_index('MachineIdentifier', inplace = True)
test.head()
index_train = list(train.index)
index_test = list(test.index)
print(len(index_train), len(index_test))
df_full = pd.concat([train, test], axis = 0)
df_full = reduce_mem_usage(df_full)
del train, test
gc.collect()
for c in df_full.columns:
    if df_full[c].dtypes == 'object':
        df_full[c] = df_full[c].astype('category')
        
df_full.info(null_counts = True)
scores = pd.DataFrame(index = ['AUC', 'LogLoss', 'Accuracy'])
imp, sc = lgbm_fast_check(df_full.loc[index_train], target_train, parameters)
scores['Source'] = sc
scores
imp.sort_values('gain', ascending = False)
gc.collect()
df_new = pd.DataFrame(index = df_full.index)
for c in version_num_features:
    print(c, time.ctime())
    df_new = pd.concat([df_new, new_features_from_version(df_full, c, target_train)], axis = 1)
df_new.head()
df_full = pd.concat([df_full, df_new], axis = 1)
gc.collect()
df_new.fillna(df_new.mean(axis = 0), inplace = True)
clusters = KMeans(n_clusters = 3, random_state = 0, n_jobs = -1)
clusters.fit(df_new.loc[index_train])
centers = clusters.cluster_centers_
gc.collect()
columns = df_new.columns
clust_features = pd.DataFrame(index = df_new.index)
for i in range(len(centers)):
    print(i, time.ctime())
    # distance as manhattan metric
    clust_features['clust_dist_' + str(i)] = (df_new[columns] - centers[i]).applymap(abs).apply(sum, axis = 1)
    
clust_features.head()
del df_new
gc.collect()
clust_features = reduce_mem_usage(clust_features)
df_full = pd.concat([df_full, clust_features], axis = 1)
df_full.head()
del clust_features
gc.collect()
df_full.info(null_counts = True)
imp, sc = lgbm_fast_check(df_full.loc[index_train], target_train, parameters)
scores['New'] = sc
scores
imp.sort_values('gain', ascending = False)
gc.collect()
scores = pd.DataFrame(index = ['auc', 'acc', 'loss', 'tn', 'fn', 'fp', 'tp'])
clfs, valid_pred, importances = lgbm_cross_validation(df_full.loc[index_train], target_train, parameters)
importances.sort_values('gain', ascending = False)
train_pred = pd.DataFrame(index = index_train)
train_pred['valid'] = valid_pred
train_pred['target'] = target_train

train_pred.head()
tn, fp, fn, tp = confusion_matrix(target_train, (train_pred['valid'] >= .5) * 1).ravel()
scores['train valid'] = [roc_auc_score(target_train, train_pred['valid']), 
                         accuracy_score(target_train, (train_pred['valid'] >= .5) * 1), 
                         log_loss(target_train, train_pred['valid']),
                         tn, fn, fp, tp]
    
scores = scores.T
scores
score_auc = scores.loc['train valid', 'auc']
score_acc = scores.loc['train valid', 'acc']
print(score_auc, score_acc)
df_full = df_full.loc[index_test]
del train_pred, target_train
gc.collect()
test_pred = pd.DataFrame(index = index_test)

for i, clf in enumerate(clfs):
    print(i, time.ctime())
    test_pred[i] = clf.predict(df_full)
    
test_pred['mean'] = test_pred.mean(axis = 1)
test_pred.head()
col = 'mean'
submit = test_pred[col].reset_index()
submit.columns = ['MachineIdentifier', 'HasDetections']
submit.head()
filename = 'subm_lgbm_{:.4f}_{:.4f}_{}fold_{}.csv'.format(score_auc, score_acc, len(clfs), 
                                                          datetime.now().strftime('%Y-%m-%d'))
print(filename)
submit.to_csv(filename, index = False)