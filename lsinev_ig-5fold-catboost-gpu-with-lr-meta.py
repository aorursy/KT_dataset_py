import numpy as np 

import pandas as pd 

import os, sys

import time

import gc

from numba import jit

try:

    import cPickle as pickle

except:

    import pickle



# from tqdm import tqdm

from tqdm import trange

from tqdm.auto import tqdm

tqdm.pandas()



from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import roc_auc_score



from sklearn import svm

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostClassifier

from sklearn import metrics

from collections import Counter



from IPython.display import HTML

import warnings

warnings.filterwarnings("ignore")



pd.set_option('max_colwidth', 500)

pd.set_option('max_columns', 500)

pd.options.display.precision = 15

# Inspired by

# https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage



def convert_col_to_proper_int(df_col):

    col_type = df_col.values.dtype

    if ((str(col_type)[:3] == 'int') | (str(col_type)[:4] == 'uint')):

        c_min = df_col.values.min()

        c_max = df_col.values.max()

        if c_min < 0:

            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:

            # https://stackoverflow.com/a/42631281 through values is faster (test with timeit)

                df_col = pd.Series(df_col.values.astype(np.int8), name=df_col.name)

            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:

                df_col = pd.Series(df_col.values.astype(np.int16), name=df_col.name)

            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:

               df_col = pd.Series(df_col.values.astype(np.int32), name=df_col.name)

            elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:

                df_col = pd.Series(df_col.values.astype(np.int64), name=df_col.name)

        else:

            if c_max <= np.iinfo(np.uint8).max:

                df_col = pd.Series(df_col.values.astype(np.uint8), name=df_col.name)

            elif c_max <= np.iinfo(np.uint16).max:

                df_col = pd.Series(df_col.values.astype(np.uint16), name=df_col.name)

            elif c_max <= np.iinfo(np.uint32).max:

                df_col = pd.Series(df_col.values.astype(np.uint32), name=df_col.name)

            elif c_max <= np.iinfo(np.uint64).max:

                df_col = pd.Series(df_col.values.astype(np.uint64), name=df_col.name)

            

    return df_col



def convert_col_to_proper_float(df_col):

    col_type = df_col.values.dtype

    if str(col_type)[:5] == 'float':

        unique_count = len(np.unique(df_col))

        # https://stackoverflow.com/a/42631281 through values is faster (test with timeit)

        df_col_temp = pd.Series(df_col.values.astype(np.float32), name=df_col.name)

        if len(np.unique(df_col_temp)) == unique_count:

            df_col = df_col_temp

            c_min = df_col.values.min()

            c_max = df_col.values.max()

            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:

                df_col_temp = pd.Series(df_col.values.astype(np.float16), name=df_col.name)

                if len(np.unique(df_col_temp)) == unique_count:

                    df_col = df_col_temp

            

    return df_col



def gentle_reduce_mem_usage(data, verbose = True, process_objects = False, cat_level = None):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    if cat_level is not None:

        cat_level = np.round(abs(cat_level) % 1, 15)

    start_mem = data.memory_usage().sum() / 1024**2

    if verbose:

        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))



    start_time = time.time()



    for col in tqdm(data.columns, desc='columns'):

#         if verbose:

#             print(col, type(data[col]), data[col].shape, 'started at', time.ctime())

        col_type = data[col].values.dtype



        if (process_objects & (col_type == object)):

            try:

                data[col] = pd.to_numeric(data[col], downcast='float')

            except ValueError:

                try:

                    data[col] = pd.to_numeric(data[col].str.replace(',', '.'), downcast='float')

                except ValueError:

                    data[col] = pd.to_datetime(data[col], infer_datetime_format=True, errors='ignore')

            col_type = data[col].values.dtype



        if (process_objects & (col_type == object) & (cat_level is not None)):

            if len(np.unique(data[col].values)) <= cat_level*len(data[col].values):

                data[col] = data[col].astype('category')

                col_type = data[col].values.dtype



        if ((col_type != object) & (col_type != '<M8[ns]') & (col_type != '<m8[ns]')\

                & (col_type.name != 'bool') & (col_type.name != 'category') & (col_type.name != 'complex64')\

                & (col_type.name != 'complex128')):#

            c_min = data[col].values.min()

            c_max = data[col].values.max()

            if ((str(col_type)[:3] == 'int') | (str(col_type)[:4] == 'uint')):

                data[col] = convert_col_to_proper_int(data[col])

            else:

                if np.isfinite(data[col].values).all():

                    if c_min < 0:

                        if abs(data[col].values - data[col].values.astype(np.int64)).sum() < 0.01:

                            data[col] = convert_col_to_proper_int(pd.Series(data[col].values.astype(np.int64),

                                                                            name=data[col].name))

                        else:

                            data[col] = convert_col_to_proper_float(data[col])

                    else:

                        if abs(data[col].values - data[col].values.astype(np.uint64)).sum() < 0.01:

                            data[col] = convert_col_to_proper_int(pd.Series(data[col].values.astype(np.uint64),

                                                                            name=data[col].name))

                        else:

                            data[col] = convert_col_to_proper_float(data[col])

                else:

                    data[col] = convert_col_to_proper_float(data[col])



    end_time = time.time()

    end_mem = data.memory_usage().sum() / 1024**2

    if verbose:

        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))

        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        print('Done in {} seconds.'.format(end_time - start_time))



    return None
RANDOM_STATE = 2042

np.random.seed(RANDOM_STATE)
NUM_THREADS = 1
input_path = '../input/'
model_name = 'model'

model_num = 0

postfix = None
%%time

train = pd.read_csv(f'{input_path}train.csv', dtype = {'target': np.uint8})

test = pd.read_csv(f'{input_path}test.csv')

ok_cols = [col for col in train.columns if col not in ['id', 'target', 'wheezy-copper-turtle-magic']]
# from https://www.kaggle.com/cdeotte/logistic-regression-0-800

# INITIALIZE VARIABLES

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in trange(512, desc='wheezy-copper-turtle-magic value'):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    test2.reset_index(drop=True,inplace=True)

    

    skf = StratifiedKFold(n_splits=25, random_state=42)

    for train_index, val_index in skf.split(train2.iloc[:,1:-1], train2['target']):

        # LOGISTIC REGRESSION MODEL

        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)

        clf.fit(train2.loc[train_index][ok_cols],train2.loc[train_index]['target'])

        oof[idx1[val_index]] += clf.predict_proba(train2.loc[val_index][ok_cols])[:,1]

        preds[idx2] += clf.predict_proba(test2[ok_cols])[:,1]

        

# PRINT CV AUC

auc = metrics.roc_auc_score(train['target'],oof.astype(np.float32))

print('LR with interactions scores CV =',round(auc,5))



train['meta_lr'] = oof.astype(np.float32)

test['meta_lr'] = preds.astype(np.float32)
train_len = len(train)

data = pd.concat([train.drop(columns=['target']), test], ignore_index = True).reset_index(drop=True)

y = train['target']

del train, test, auc

gc.collect()

gentle_reduce_mem_usage(data)

data.info()
%%time

col_part_names = [col.split('-') for col in ok_cols]

col_part_names = [i for j in col_part_names for i in j]

some_cols = [i[0] for i in Counter(col_part_names).most_common() if i[1] > 4]



#frequencies

data['wheezy-copper-turtle-magic_count'] = data.groupby(['wheezy-copper-turtle-magic'])['id'].transform('count')



for col in tqdm(ok_cols, desc='processed columns'):

    data[f'{col}_w_mean'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform('mean').fillna(0).values.astype(np.float32)

    data[f'{col}_w_std'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform('std').fillna(0).values.astype(np.float32)

    data[f'{col}_w_max'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform('max').fillna(0).values.astype(np.float32)

    data[f'{col}_w_min'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform('min').fillna(0).values.astype(np.float32)

    data[f'{col}_w_quantile_10'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform(lambda x: np.percentile(x.unique(), 10)).fillna(0).values.astype(np.float32)

#     data[f'{col}_w_quantile_25'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform(lambda x: np.percentile(x.unique(), 25)).fillna(0).values.astype(np.float32)

#     data[f'{col}_w_median'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform(lambda x: np.percentile(x.unique(), 50)).fillna(0).values.astype(np.float32)

#     data[f'{col}_w_quantile_75'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform(lambda x: np.percentile(x.unique(), 75)).fillna(0).values.astype(np.float32)

    data[f'{col}_w_quantile_90'] = data.groupby(['wheezy-copper-turtle-magic'])[col].transform(lambda x: np.percentile(x.unique(), 90)).fillna(0).values.astype(np.float32)



for c in tqdm(some_cols, desc='some_cols columns'):

    more_such_cols = [col for col in ok_cols if c in col]

    data[f'{c}_mean'] = data[more_such_cols].mean(1)

    data[f'{c}_min'] = data[more_such_cols].min(1)

    data[f'{c}_max'] = data[more_such_cols].max(1)

    data[f'{c}_std'] = data[more_such_cols].std(1)



scaler = StandardScaler()

print('Scaling with StandardScaler\n')

data[ok_cols] = scaler.fit_transform(data[ok_cols])
data.shape
gentle_reduce_mem_usage(data)

data.info()
# data['wheezy-copper-turtle-magic'] = data['wheezy-copper-turtle-magic'].astype(object) #making cat_feature for catboost

X = data[:train_len].drop(['id'], axis=1)

X_test = data[train_len:].drop(['id'], axis=1)

X.shape
X_test.shape
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)
del data

gc.collect()
@jit

def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True





def train_model_classification(X, X_test, y, params, folds, model_type='cat', eval_metric='auc', columns=None,

                               plot_feature_importance=False, model=None, cat_plot = None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of classification models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',#AUC:hints=skip_train~false

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    'mcc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'MCC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    loss_dict ={'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'Logloss',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                'mcc': {

                    'lgb_metric_name': eval_auc,

                    'catboost_metric_name': 'Logloss',

                    'sklearn_scoring_function': metrics.roc_auc_score},

                }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros((len(X), len(set(y.values))))

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test), oof.shape[1]))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = NUM_THREADS)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)],

                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, n_jobs = NUM_THREADS,

                              early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, thread_count=NUM_THREADS,

                                       early_stopping_rounds = early_stopping_rounds,

                                       eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                       loss_function=loss_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),

                      plot = cat_plot,

                      use_best_model=True, verbose=verbose)



            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test)

        

#         print(oof[valid_index].shape)

#         print(y_pred_valid.shape)

        oof[valid_index] = y_pred_valid

        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid[:, 1]))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f} (CV score: {0:.3f}±{2:.3f}).'.format(np.mean(scores), np.std(scores),

                                                                                     3*np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict

params = {'metric_period': 200, #auc calculation is slow, so evaluate less

#           'cat_features': [X.columns.get_loc('wheezy-copper-turtle-magic')],

          'random_seed': RANDOM_STATE,

          'depth': 8,

          'bagging_temperature': 0.825,

          'random_strength': 0.125,

#           'l2_leaf_reg': 6.0,

          'task_type': 'GPU',

#           'learning_rate': 0.10,

          'od_type': 'Iter',

          'custom_metric': 'MCC',

         }

result_dict_cat = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='cat',

                                             cat_plot = False,

                                             eval_metric='auc', plot_feature_importance=True, verbose=200, n_estimators=30000)
result_dict_cat['prediction'].shape
result_dict_cat['oof'].shape
sub = pd.read_csv(f'{input_path}sample_submission.csv')

sub['target'] = result_dict_cat['prediction'][:, 1]

sub.to_csv("submission.csv", index=False)

# sub.head()