%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from tqdm._tqdm_notebook import tqdm_notebook as tqdm
# tqdm.pandas()
from tqdm import tqdm

sns.set_style('white')
gc.enable()

DATA_DIR = '../input/home-credit-default-risk/'

import glob
def get_path(str, first=True, parent_dir='../input/**/'):
    res_li = glob.glob(parent_dir+str)
    return res_li[0] if first else res_li

def load_folds_lables():
    path = '../input/hcdr-prepare-kfold/'
    eval_sets = np.load(path+'eval_sets.npy')
    y = np.load(path+'target.npy')
    return eval_sets, y
folds, labels = load_folds_lables()
nfolds = 5
train_num = len(labels)
# nrows = None
# bureau = pd.read_csv(DATA_DIR+'bureau.csv', nrows=nrows)
# bureau_balance = pd.read_csv(DATA_DIR+'bureau_balance.csv', nrows=nrows)
# previous_application = pd.read_csv(DATA_DIR+'previous_application.csv', nrows=nrows)
# installments_payments = pd.read_csv(DATA_DIR+'installments_payments.csv', nrows=nrows)
# credit_card_balance = pd.read_csv(DATA_DIR+'credit_card_balance.csv', nrows=nrows)
# pos_cash_balance = pd.read_csv(DATA_DIR+'POS_CASH_balance.csv', nrows=nrows)
# train = pd.read_csv(DATA_DIR+'application_train.csv', usecols=lambda c:c!='TARGET')
# test = pd.read_csv(DATA_DIR+'application_test.csv')

# data = feather.read_dataframe(get_path('data.ftr'))
# train = data[:train_num]
# test = data[train_num:].reset_index(drop=True)
# del data; gc.collect();
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def nest_print(dict_item, inline=True, indent=True):
    s = []
    s_ind = '\t' if indent else ''
    for k, v in dict_item.items():
        s += [': '.join([str(k), str(round(v, 6))])]
    if inline:
        print(s_ind+' '.join(s))
    else:
        print(s_ind+'\n'.join(s))

def lgb_cv_train(
    name, params, X, y, X_test, feature_name,
    num_boost_round, early_stopping_rounds, verbose_eval,
    cv_folds, metric=roc_auc_score,
    verbose_cv=True, nfolds=nfolds, msgs={}
):
    pred_test = np.zeros((X_test.shape[0],))
    pred_val = np.zeros((X.shape[0],))
    cv_scores = []
    feat_imps = []
    models = []
    for valid_fold in range(nfolds):
        mask_te = cv_folds==valid_fold
        mask_tr = ~mask_te
        print('[level 1] processing fold %d...'%(valid_fold+1))
        t0 = time.time()
        dtrain = lgb.Dataset(
            X[mask_tr], y[mask_tr],
            feature_name=feature_name,
            free_raw_data=False
        )
        dvalid = lgb.Dataset(
            X[mask_te], y[mask_te],
            feature_name=feature_name,
            free_raw_data=False
        )
        evals_result = {}
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=['train','valid'],
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        pred_val[mask_te] = model.predict(X[mask_te])
        pred_test += model.predict(X_test)/nfolds
        scr = metric(y[mask_te], pred_val[mask_te])
        feat_imps.append(model.feature_importance()/model.best_iteration)
        if verbose_cv:
            print(f'{name} auc:', scr, 
                  f'fold {valid_fold+1} done in {time.time() - t0:.2f} s')
        cv_scores.append(scr)
        models.append(model)
    msgs = dict(
        msgs, 
        cv_score_mean=np.mean(cv_scores), 
        cv_score_std=np.std(cv_scores),
        cv_score_min=np.min(cv_scores), 
        cv_score_max=np.max(cv_scores),
    )
    nest_print(msgs)
    result = dict(
        name=name,
        pred_val=pred_val,
        pred_test=pred_test,
        cv_scores=cv_scores,
        models=models,
        feat_imps=feat_imps
    )
    return result
train_ids = pd.read_csv(get_path('*application_train.csv'), usecols=['SK_ID_CURR'])['SK_ID_CURR'].values
test_ids = pd.read_csv(get_path('*application_test.csv'), usecols=['SK_ID_CURR'])['SK_ID_CURR'].values
sk_id_curr = np.load(get_path('sk_id_curr*'))
train_ids.shape, test_ids.shape, sk_id_curr.shape
def label_encoding(df):
    obj_cols = [c for c in df.columns if df[c].dtype=='O']
    for c in obj_cols:
        df[c] = pd.factorize(df[c], na_sentinel=-1)[0]
    df[obj_cols].replace(-1, np.nan, inplace=True)
    return df
id_labels = pd.DataFrame()
id_labels['SK_ID_CURR'] = sk_id_curr
id_labels['TARGET'] = -1
id_labels['TARGET'][:train_num] = labels
id_labels['fold'] = -1
id_labels['fold'][:train_num] = folds
results = {}

lgb_params =  {
    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc', 
    'num_threads': 4, 
    #'min_data_in_leaf': 50, #20
    'max_depth': 8, #10, 
    'num_leaves': 32,
    'seed': 233,
    'lambda_l1': 0.04,
    'lambda_l2': 0.04,
    'feature_fraction': 0.9497036, #0.7
    'bagging_fraction': 0.8715623, #0.7
    'bagging_freq': 1, #4
    'learning_rate': 0.03, #0.016
    'min_split_gain': 0.0222415,
    'min_child_weight': 40,
    'verbose': -1
}

round_params = dict(
    num_boost_round = 20000,
    early_stopping_rounds = 50,
    verbose_eval = 50,
)
# csvname = 'bureau'
csvname_li = [
    'bureau',
    #'bureau_balance',
    'previous_application',
    'installments_payments',
    'credit_card_balance',
    'POS_CASH_balance',
]

for csvname in csvname_li:
    print(f'Current: {csvname}...')
    
    df = pd.read_csv(DATA_DIR+f'{csvname}.csv')
    df = df.loc[np.isin(df['SK_ID_CURR'], sk_id_curr)]
    df = df.merge(id_labels, how='left', on='SK_ID_CURR')
    df = label_encoding(df)
    eval_cols = ['SK_ID_CURR', 'fold', 'TARGET']
    eval_df = df[eval_cols].copy()
    df.drop(eval_cols, axis=1, inplace=True)

    feature_name = df.columns.tolist()
    y = eval_df['TARGET'].values.copy()
    X = df.loc[y!=-1].values
    X_test = df.loc[y==-1].values
    cv_folds = eval_df.loc[y!=-1, 'fold'].values
    y = y[y!=-1]

    print('shapes', X.shape, y.shape, cv_folds.shape, X_test.shape)

    results[csvname] = lgb_cv_train(
        f'{csvname}', lgb_params,
        X, y, X_test, feature_name, 
        cv_folds=cv_folds,
        **round_params
    )

    eval_df['pred'] = -1
    eval_df.loc[eval_df['fold']!=-1, 'pred'] = results[csvname]['pred_val']
    eval_df.loc[eval_df['fold']==-1, 'pred'] = results[csvname]['pred_test']
    results[csvname]['eval_df'] = eval_df.copy()
from joblib import Parallel, delayed
def calc_stats(key, group):
    res = pd.Series()
    res['SK_ID_CURR'] = key
    res['_max'] = np.max(group)
    res['_min'] = np.min(group)
    res['_median'] = np.median(group)
    res['_mean'] = np.mean(group)
    res['_std'] = np.std(group)
    res['_size'] = np.size(group)
    res['_sum'] = np.sum(group)
    res['_skew'] = group.skew()
    res['_kurtosis'] = group.kurtosis()
    return res
stat_func_li = ['size', 'sum', 'min', 'median', 'max', 'mean', 'std', 'skew', 'kurtosis']
import pickle
with open('mr_results.pkl', 'wb') as f:
    pickle.dump(results, f)
pred_stats = pd.DataFrame()
pred_stats['SK_ID_CURR'] = sk_id_curr
for csvname in csvname_li:
    print(f'Calculating pred stats of {csvname}')
    grp = results[csvname]['eval_df'].groupby('SK_ID_CURR')
    res = Parallel(n_jobs=-1)(
        delayed(calc_stats)(key, group) for key, group in tqdm(grp['pred'], total=len(grp.groups))
    )
    res = pd.concat(res, axis=1).T
    res['SK_ID_CURR'] = res['SK_ID_CURR'].astype('int')
    res.columns = [csvname+c if c!='SK_ID_CURR' else c  for c in res.columns]
    pred_stats = pred_stats.merge(res, on='SK_ID_CURR', how='left')
pred_stats.to_csv('pred_stats.csv', index=False)
del pred_stats['SK_ID_CURR']