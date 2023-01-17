# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
df_test = pd.read_csv("../input/application_test.csv")
df_train = pd.read_csv("../input/application_train.csv")
### Code modified from Aguiar's kernel: https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
def kfold_lightgbm(df, num_folds=5, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        # For best performance 10000 estimators, use fewer to speed up prediction time
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    test_df['TARGET'] = sub_preds
    display_importances(feature_importance_df)
    return test_df, feature_importance_df

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
def installments_payments(df):
    aggs = ['sum', 'mean', 'max', 'min']
    feat_cols = ['NUM_INSTALMENT_NUMBER', 'AMT_INSTALMENT', 'AMT_PAYMENT']
    
    agg_dict = {k:aggs for k in feat_cols}
    df_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    
    df_agg.columns = [col + '_' + agg.upper() + '_INSTALL' for col in feat_cols for agg in agg_dict[col]]
    return df_agg

def pos_cash_balance(df):
    df['COMPLETED_CONTRACTS'] = (df['NAME_CONTRACT_STATUS'] == 'Active').astype(int)
    
    aggs = ['sum', 'mean']
    feat_cols = ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'COMPLETED_CONTRACTS']
    
    agg_dict = {k:aggs for k in feat_cols}
    agg_dict['COMPLETED_CONTRACTS'] = ['mean', 'size']
    df_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    
    df_agg.columns = [col + '_' + agg.upper() + '_POS' for col in feat_cols for agg in agg_dict[col]]
    return df_agg

def credit_card_balance(df):
    df['PCT_CREDIT_LIMIT'] = df['AMT_BALANCE']/df['AMT_CREDIT_LIMIT_ACTUAL']
    
    aggs = ['sum', 'mean', 'max']
    feat_cols = ['PCT_CREDIT_LIMIT', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 
                 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
                 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
                 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT',
                 'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD']
    
    agg_dict = {k:aggs for k in feat_cols}
    agg_dict['PCT_CREDIT_LIMIT'] = ['mean', 'max']
    df_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    
    df_agg.columns = [col + '_' + agg.upper() + '_CREDIT' for col in feat_cols for agg in agg_dict[col]]
    return df_agg

def bureau(df):
    df['CREDIT_ACTIVE_NUM'] = (df.CREDIT_ACTIVE == 'ACTIVE').astype(int)
    
    aggs = ['sum', 'mean', 'max']
    feat_cols = ['CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_MAX_OVERDUE',
                 'AMT_CREDIT_SUM',
                 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY',
                 'CREDIT_ACTIVE_NUM', 'AMT_CREDIT_SUM_DEBT']
    
    agg_dict = {k:aggs for k in feat_cols}
    agg_dict['CREDIT_ACTIVE_NUM'] = ['sum', 'mean', 'size']
    df_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    
    df_agg.columns = [col + '_' + agg.upper() + '_BUREAU' for col in feat_cols for agg in agg_dict[col]]
    return df_agg

def application_train(df):
    feat_num_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                     'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
                     'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS',
                     'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    feat_binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    
    for col in feat_binary_cols:
        df[col] = df[col].factorize()[0]
    return df[['SK_ID_CURR', 'TARGET'] + feat_num_cols + feat_binary_cols].set_index('SK_ID_CURR')

def previous_application(df):
    df['CONTRACTS_REFUSED'] = (df['NAME_CONTRACT_STATUS'] == 'REFUSED').astype(int)
    
    aggs = ['sum', 'mean', 'max']
    feat_cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
                     'AMT_GOODS_PRICE', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
                     'RATE_INTEREST_PRIVILEGED', 'CNT_PAYMENT', 'CONTRACTS_REFUSED']
    
    agg_dict = {k:aggs for k in feat_cols}
    df_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    
    df_agg.columns = [col + '_' + agg.upper() + '_PREV' for col in feat_cols for agg in aggs]
    return df_agg
    
###Combines helper functions to output prediction
def crossval_predict(df_train, df_test, fit_predictor, 
                     previous_app_features=True,
                     bureau_features=True,
                     credit_features=True,
                     pos_features=True,
                     install_features=True):
    hold = []
    
    hold.append(application_train(pd.concat([df_train, df_test])))
    del(df_train, df_test)
    gc.collect()
    
    if previous_app_features:
        df_prev = pd.read_csv('../input/previous_application.csv')
        hold.append(previous_application(df_prev))
        del(df_prev)
        gc.collect()
    
    if bureau_features:
        df_bureau = pd.read_csv('../input/bureau.csv')
        hold.append(bureau(df_bureau))
        del(df_bureau)
        gc.collect()
    
    if credit_features:
        df_credit = pd.read_csv('../input/credit_card_balance.csv')
        hold.append(credit_card_balance(df_credit))
        del(df_credit)
        gc.collect()
    
    if pos_features:
        df_pos = pd.read_csv('../input/POS_CASH_balance.csv')
        hold.append(pos_cash_balance(df_pos))
        del(df_pos)
        gc.collect()
    
    if install_features:
        df_install = pd.read_csv('../input/installments_payments.csv')
        hold.append(installments_payments(df_install))
        del(df_install)
        gc.collect()
    
    df_test, feat_importance = fit_predictor(pd.concat(hold, axis=1))
    del(hold)
    gc.collect()
    
    return df_test[['TARGET']].reset_index()

%%time
output = crossval_predict(df_train, 
                          df_test,
                          kfold_lightgbm)
output.to_csv('extratrees_simple.csv', index=False)
