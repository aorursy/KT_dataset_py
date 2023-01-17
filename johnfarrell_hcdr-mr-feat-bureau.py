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
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()

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

def get_imp_plot(lgb_res, savefig=True):
    lgb_feat_imps = lgb_res['feat_imps']
    name = lgb_res['name']
    lgb_imps = pd.DataFrame(
        np.vstack(lgb_feat_imps).T, 
        columns=['fold_{}'.format(i) for i in range(nfolds)],
        index=feature_name,
    )
    lgb_imps['fold_mean'] = lgb_imps.mean(1)
    lgb_imps = lgb_imps.loc[
        lgb_imps['fold_mean'].sort_values(ascending=False).index
    ]
    lgb_imps.reset_index().to_csv(f'{name}_lgb_imps.csv', index=False)
    del lgb_imps['fold_mean']; gc.collect();

    max_num_features = min(len(feature_name), 300)
    f, ax = plt.subplots(figsize=[8, max_num_features//4])
    data = lgb_imps.iloc[:max_num_features].copy()
    data_mean = data.mean(1).sort_values(ascending=False)
    data = data.loc[data_mean.index]
    data_index = data.index.copy()
    data = [data[c].values for c in data.columns]
    data = np.hstack(data)
    data = pd.DataFrame(data, index=data_index.tolist()*nfolds, columns=['igb_imp'])
    data = data.reset_index()
    data.columns = ['feature_name', 'igb_imp']
    sns.barplot(x='igb_imp', y='feature_name', data=data, orient='h', ax=ax)
    plt.grid()
    if savefig:
        plt.savefig(f'{name}_lgb_imp.png')

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
    'min_data_in_leaf': 80,
    'max_depth': -1,
    'num_leaves': 28,
    'seed': 233,
    'lambda_l1': 0.04,
    'lambda_l2': 0.4,
    'feature_fraction': 0.84,
    'bagging_fraction': 0.77,
    'bagging_freq': 1,
    'learning_rate': 0.02,
    #'min_split_gain': 0.0222415,
    #'min_child_weight': 40,
    'verbose': -1
}

round_params = dict(
    num_boost_round = 20000,
    early_stopping_rounds = 100,
    verbose_eval = 50,
)
csvname_li = [
    'bureau',
    #'bureau_balance',
    'previous_application',
    'installments_payments',
    'credit_card_balance',
    'POS_CASH_balance',
]

csvname = 'bureau'

print(f'Current: {csvname}...')

df = pd.read_csv(DATA_DIR+f'{csvname}.csv')
df = df.loc[np.isin(df['SK_ID_CURR'], sk_id_curr)]
df = df.merge(id_labels, how='left', on='SK_ID_CURR')
df = label_encoding(df)
df.head().T
df['days_end_fact_diff'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT_ENDDATE']
df['days_upd_fact_diff'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT_UPDATE']
df['days_crd_fact_diff'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT']

df['day_ovd_prl_ratio'] = df['CREDIT_DAY_OVERDUE'] / (1+df['CNT_CREDIT_PROLONG'])
df['amt_ovd_prl_ratio'] = df['AMT_CREDIT_SUM_OVERDUE'] / (1+df['CNT_CREDIT_PROLONG'])
df['amt_day_ovd_ratio'] = df['AMT_CREDIT_SUM_OVERDUE'] / (1+df['CREDIT_DAY_OVERDUE'])
df['amt_max_ovd_prl_ratio'] = df['AMT_CREDIT_MAX_OVERDUE'] / (1+df['CNT_CREDIT_PROLONG'])
df['amt_max_day_ovd_ratio'] = df['AMT_CREDIT_MAX_OVERDUE'] / (1+df['CREDIT_DAY_OVERDUE'])
df['amt_debt_sum_ratio'] = df['AMT_CREDIT_SUM_DEBT'] / (1+df['AMT_CREDIT_SUM'])
df['amt_max_ovd_sum_ratio'] = df['AMT_CREDIT_MAX_OVERDUE'] / (1+df['AMT_CREDIT_SUM'])
df['amt_max_ovd_sum_ratio'] = df['AMT_ANNUITY'] / (1+df['AMT_CREDIT_SUM'])

df['amt_ann_debt_ratio'] = df['AMT_ANNUITY'] / (1+df['AMT_CREDIT_SUM_DEBT'])
df['amt_debt_prl_prod'] = df['AMT_CREDIT_SUM_DEBT'] * df['CNT_CREDIT_PROLONG']
df['amt_ann_prl_prod'] = df['AMT_ANNUITY'] * df['CNT_CREDIT_PROLONG']
df['day_ovd_prl_prod'] = df['CREDIT_DAY_OVERDUE'] * df['CNT_CREDIT_PROLONG']
bb = pd.read_csv(DATA_DIR+'bureau_balance.csv')
bb = pd.concat([bb, pd.get_dummies(bb['STATUS'], prefix='bb_status')], axis=1)
del bb['STATUS']
bb_grp = bb.groupby('SK_ID_BUREAU', sort=False)
cols = [c for c in bb.columns if 'bb_status_' in c]
bb_feat = pd.DataFrame()
bb_feat = bb_grp[cols].agg({c: ['mean', 'sum'] for c in cols})
bb_feat['bb_size'] = bb_grp.size()
bb_feat.columns = ['_'.join(_) for _ in bb_feat.columns.tolist()]
bb_feat = bb_feat.reset_index()
df = df.merge(bb_feat, how='left', on='SK_ID_BUREAU')
del bb_feat; gc.collect();
from joblib import Parallel, delayed
def func(index, group, main_key, prefix=''):
    res = pd.Series()
    
    res['amt_max_ovd_mean'] = np.log1p(group['AMT_CREDIT_MAX_OVERDUE']).mean()
    res['amt_sum_ovd_mean'] = np.log1p(group['AMT_CREDIT_SUM_OVERDUE']).mean()
    res['amt_sum_mean'] = np.log1p(group['AMT_CREDIT_SUM']).mean()
    res['amt_debt_mean'] = np.log1p(group['AMT_CREDIT_SUM_DEBT']).mean()
    res['amt_debt_sum_ratio_mean'] = group['amt_debt_sum_ratio'].mean()
    res['day_ovd_mean'] = group['CREDIT_DAY_OVERDUE'].mean()
    res['day_ef_diff_mean'] = group['days_end_fact_diff'].mean()
    res['prl_mean'] = group['CNT_CREDIT_PROLONG'].mean()
    
    res['amt_max_ovd_std'] = np.log1p(group['AMT_CREDIT_MAX_OVERDUE']).std()
    res['amt_sum_ovd_std'] = np.log1p(group['AMT_CREDIT_SUM_OVERDUE']).std()
    res['amt_sum_std'] = np.log1p(group['AMT_CREDIT_SUM']).std()
    res['amt_debt_std'] = np.log1p(group['AMT_CREDIT_SUM_DEBT']).std()
    res['amt_debt_sum_ratio_std'] = group['amt_debt_sum_ratio'].std()
    res['day_ovd_std'] = group['CREDIT_DAY_OVERDUE'].std()
    res['day_ef_diff_std'] = group['days_end_fact_diff'].std()
    res['prl_std'] = group['CNT_CREDIT_PROLONG'].std()
    
    if prefix!='':
        res.index = [prefix+c for c in res.index]
    
    res[main_key] = index
    return res
%%time
main_key = 'CREDIT_TYPE'
feat = Parallel(n_jobs=8)(
    delayed(func)(key, group, main_key, 'credit_type_') for key,group in tqdm(
        df.groupby([main_key], sort=False)
    )
)
feat = pd.DataFrame(feat)
df = df.merge(feat, on=main_key, how='left')
%%time
df['bb_size_'].fillna(-1, inplace=True)
main_key = 'bb_size_'
feat = Parallel(n_jobs=8)(
    delayed(func)(key, group, main_key, 'bb_size_') for key,group in tqdm(
        df.groupby([main_key], sort=False)
    )
)
feat = pd.DataFrame(feat)
df = df.merge(feat, on=main_key, how='left')
df['bb_size_'].replace(-1, np.nan, inplace=True)
%%time
main_key = 'CREDIT_ACTIVE'
feat = Parallel(n_jobs=8)(
    delayed(func)(key, group, main_key, 'credit_act_') for key,group in tqdm(
        df.groupby([main_key], sort=False)
    )
)
feat = pd.DataFrame(feat)
df = df.merge(feat, on=main_key, how='left')
eval_cols = ['SK_ID_CURR', 'fold', 'TARGET']
eval_df = df[eval_cols].copy()
df.drop(eval_cols+['SK_ID_BUREAU'], axis=1, inplace=True)

feature_name = df.columns.tolist()
y = eval_df['TARGET'].values.copy()
X = df.loc[y!=-1].values
X_test = df.loc[y==-1].values
cv_folds = eval_df.loc[y!=-1, 'fold'].values
y = y[y!=-1]

del df; gc.collect();
print('shapes', X.shape, y.shape, cv_folds.shape, X_test.shape)
res = lgb_cv_train(
    f'{csvname}', lgb_params,
    X, y, X_test, feature_name, 
    cv_folds=cv_folds,
    **round_params
)

eval_df['pred'] = -1
eval_df.loc[eval_df['fold']!=-1, 'pred'] = res['pred_val']
eval_df.loc[eval_df['fold']==-1, 'pred'] = res['pred_test']
eval_df.to_csv('eval_res.csv', index=False)
get_imp_plot(res, False)