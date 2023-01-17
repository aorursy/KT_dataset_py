'''
LGBM in iteration
'''
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
def group(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()

def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)
def POS_CASH_balance():
    pos = pd.read_pickle(f'../input/preload-data/POS_CASH_balance.pkl')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    del pos
    gc.collect()
    return pos_agg
def credit_card_balance():
    cc = pd.read_pickle(f'../input/preload-data/credit_card_balance.pkl')
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc['LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc_=cc.copy()
    cc_.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc_.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc_.groupby('SK_ID_CURR').size()

    CREDIT_CARD_AGG = {
        'CNT_DRAWINGS_ATM_CURRENT': ['mean'],
        'SK_DPD': ['max', 'sum'],
        'AMT_BALANCE': ['mean', 'max'],
        'LIMIT_USE': ['max', 'mean']
    }

    # Aggregations for last x months
    for months in [12, 24, 48]:
        cc_prev_id = cc[cc['MONTHS_BALANCE'] >= -months]['SK_ID_PREV'].unique()
        cc_recent = cc[cc['SK_ID_PREV'].isin(cc_prev_id)]
        prefix = 'INS_{}M_'.format(months)
        cc_agg = group_and_merge(cc_recent, cc_agg, prefix, CREDIT_CARD_AGG)

    del cc,cc_
    gc.collect()
    return cc_agg
def installments_payments():
    ins = pd.read_pickle(f'../input/preload-data/installments_payments.pkl')
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    ins['DPD_7'] = ins['DPD'].apply(lambda x: 1 if x >= 7 else 0)
    ins['DPD_15'] = ins['DPD'].apply(lambda x: 1 if x >= 15 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'SK_ID_PREV': ['size', 'nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'DPD_7': ['mean'],
        'DPD_15': ['mean'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['min','max', 'mean', 'sum']
        }


    INSTALLMENTS_AGG = {
        'SK_ID_PREV': ['size'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DPD': ['max', 'mean', 'var'],
        'DBD': ['max', 'mean', 'var'],
        'PAYMENT_PERC': ['mean'],
        'PAYMENT_DIFF': ['mean'],
        'DPD_7': ['mean'],
        'DPD_15': ['mean'],
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    for months in [36, 60]:
        recent_prev_id = ins[ins['DAYS_INSTALMENT'] >= -30*months]['SK_ID_PREV'].unique()
        pay_recent = ins[ins['SK_ID_PREV'].isin(recent_prev_id)]
        prefix = 'INSTAL_{}M_'.format(months)
        ins_agg = group_and_merge(pay_recent, ins_agg, prefix, INSTALLMENTS_AGG)
    
    del ins
    gc.collect()
    return ins_agg
def previous_application():
    prev = pd.read_pickle( f'../input/preload-data/previous_application.pkl')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    gc.collect()
    return prev_agg
# https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    '''
    df: Final DataFrame of shape (n_samples, n_features)
    '''
    # Divide into training/validation and test data
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
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
    # features & determine feature importances
    feature_importance_df = pd.DataFrame()
    feats = train_df.columns.drop(['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV'],errors='ignore')
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        params = {
            'num_iterations':10000, 'n_iter_no_change':100,
            'learning_rate': 0.02, 'max_depth': 8, 'num_leaves': 31, 'reg_alpha': 0.37, 'reg_lambda': 0.56,
            'metric':'auc', 
            'verbose':-1, 
            'random_state':0
        }
        clf = LGBMClassifier(
            nthread=4,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            **params,
        )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            verbose=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        # append below
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['TARGET']].to_csv("submission.csv", index=True)
    return feature_importance_df

def display_importances(feature_importance_df):
    mean_df = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
    mean_df.to_csv('mean_importance.csv')
    # top 40, with error bar
    best_features = mean_df[:40].index
    best_df = feature_importance_df.loc[feature_importance_df.feature.isin(best_features)]
    plt.figure(figsize=(16, 20))
    sns.barplot(x="importance", y="feature", order=best_features, data=best_df)
    plt.title('LightGBM Top 40 Features (avg over folds)')
    plt.savefig('LightGBM Top 40 Features.png')
    # almost of no importance
    print("almost of no importance: \n",mean_df[mean_df['importance']<1.0].index)
    print(mean_df[mean_df['importance']<1.0])
    
def application(df):
    '''
    df: merged
    '''

    # drop feature of no importance
#     no_importance = ['FLAG_DOCUMENT_21', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_DOCUMENT_9',
#         'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_20',
#         'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_19',
#         'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_12',
#         'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_13']
#     df.drop(columns=no_importance,inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # inspired from https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
    df['INCOME_GOODS_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_GOODS_PRICE']
    df['CREDIT_GOODS_PERC'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['REGION_POPULATION_RELATIVE'] = df['REGION_POPULATION_RELATIVE'].astype('category')
    df['EXT1_3_PERC'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_3']
    df['EXT2_3_PERC'] = df['EXT_SOURCE_2'] / df['EXT_SOURCE_3']
    df['EXT3_SQ'] = df['EXT_SOURCE_3'] ** 2
    df['REQH_EXT3_PERC'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] / df['EXT_SOURCE_3']
    df['REQD_EXT3_PERC'] = df['AMT_REQ_CREDIT_BUREAU_DAY'] / df['EXT_SOURCE_3']
    df['REQW_EXT3_PERC'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'] / df['EXT_SOURCE_3']
    df['REQM_EXT3_PERC'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / df['EXT_SOURCE_3']
    df['REQQ_EXT3_PERC'] = df['AMT_REQ_CREDIT_BUREAU_QRT'] / df['EXT_SOURCE_3']
    df['DEF30_PERC'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
    df['DEF60_PERC'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE']
    
    return df
def time_weighted_agg(df, idxcol, tmcol, fcols):
    '''
    return: DataFrame with tmcol, fcols aggregated 
    '''
    # weights
    df[tmcol] = df[tmcol].transform(np.exp2, dtype='float32')
    # normalize the observations
    for col in fcols:
        df[col] *= df[tmcol] 
    g = df.groupby(idxcol)
    # grouped sum of weighted observations
    df = g[fcols].agg('sum')
    # grouped sum of weights
    s = g[tmcol].agg('sum')
    # weighted avg
    df = df.div(s,axis='index')
    df.rename(columns={col: col + "_wavg" for col in fcols}, inplace=True) 
    
    return df


def bureau_and_balance(bureau, bb):  
    # bb
    bb = pd.get_dummies(bb, dummy_na=False) # STATUS unknown expressed as X
    bb_cat = bb.columns[bb.columns.str.contains(r"^STATUS_")]
    # part I: agg regardless of time
    bb1_agg = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    # STATUS record: stain
    for cat in bb_cat: 
        bb1_agg[cat] = ['max', 'sum']
    bb1 = bb.groupby('SK_ID_BUREAU').agg(bb1_agg)
    # flatten multiindex of the form ( , )
    bb1.columns = pd.Index([tup[0] + "_" + tup[1].upper() for tup in bb1.columns])
    # part II
    bb2 = time_weighted_agg(bb, idxcol='SK_ID_BUREAU', tmcol='MONTHS_BALANCE',fcols=bb_cat)
    
    bb = bb1.join(bb2,on='SK_ID_BUREAU')
    
    # bureau
    bureau = bureau.join(bb,on='SK_ID_BUREAU')
    bureau.drop(columns='SK_ID_BUREAU',inplace=True)
    # percentages
    bureau['CREDIT_PERC'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['DEBT_PERC'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['MAXOVERDUE_PERC'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['OVERDUE_PERC'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['ANNUITY_LIMIT_PERC'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['ANNUITY_DEBT_PERC'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM_DEBT']
    bureau['ANNUITY_CREDIT_PERC'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']    
    # cat -> num using ohe
    bureau = pd.get_dummies(bureau, dummy_na=False)
    # prepare to aggregate
    idxcol = 'SK_ID_CURR'
    # part I: agg regardless of time
    bu1_col = bureau.select_dtypes(include='number').columns.drop(idxcol)
    bu1_agg = {
        'DAYS_CREDIT': ['min', 'max'],
        'DAYS_ENDDATE_FACT': ['min', 'max'],
        'DAYS_CREDIT_UPDATE': ['max', 'size'],
    }
    for col in bu1_col:
        if col not in bu1_agg:
            bu1_agg[col] = ['max','sum']
    bu1 = bureau.groupby(idxcol).agg(bu1_agg)
    # flatten multiindex of the form ( , )
    bu1.columns = pd.Index([tup[0] + "_" + tup[1].upper() for tup in bu1.columns])
    
    # part II
    tmcol = 'MONS_CREDIT_UPDATE' # negative relative time in MONS
    bureau.rename(columns={'DAYS_CREDIT_UPDATE': tmcol},inplace=True)
    bureau[tmcol] = (bureau[tmcol] - bureau[tmcol].max()) / 32
    # drop other time feat
    bu2 = bureau.drop(columns=['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT'])
    # aggregate
    fcols = bureau.select_dtypes(include='number').columns.drop([idxcol, tmcol])
    bu2 = time_weighted_agg(bureau, idxcol, tmcol, fcols)
    
    bureau = bu1.join(bu2, on=idxcol)
    bureau = bureau.add_prefix('bu_') # add prefix
    
    return bureau
def main(debug = False):
    df = pd.read_pickle(f'../input/preload-data/application_train.pkl')
    test_df = pd.read_pickle(f'../input/preload-data/application_test.pkl')
    bureau = pd.read_pickle(f'../input/preload-data/bureau.pkl')
    bb = pd.read_pickle(f'../input/preload-data/bureau_balance.pkl')
    if debug:
        df = df.sample(frac=0.05, random_state=0)
        test_df = test_df.sample(frac=0.05, random_state=0)
        bureau = bureau.sample(frac=0.05, random_state=0)
        bb = bb.sample(frac=0.05, random_state=0)
    df = df.append(test_df)
    # kinda align
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    df = application(df).set_index('SK_ID_CURR',verify_integrity=True)

    bu_agg = bureau_and_balance(bureau, bb)
    
    # cxy
    pos_agg=POS_CASH_balance()
    cc_agg=credit_card_balance().set_index('SK_ID_CURR',verify_integrity=True)
    ins_agg=installments_payments().set_index('SK_ID_CURR',verify_integrity=True)
    prev_agg=previous_application()
    
    # Final DataFrame
    df = df.join([bu_agg, pos_agg, cc_agg, ins_agg, prev_agg])
    df.to_pickle(f'final_df.pkl')
    
    feature_importance_df = kfold_lightgbm(df, 5, stratified = False, debug= False)
    display_importances(feature_importance_df)

main()