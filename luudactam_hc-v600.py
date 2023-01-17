import pandas as pd
pd.options.display.max_columns = 999
import numpy as np
import warnings
from sklearn.model_selection import train_test_split,  KFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
train = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")
previous = pd.read_csv("../input/previous_application.csv")
#install = pd.read_csv("../input/installments_payments.csv")

bureau = pd.read_csv("../input/bureau.csv")
previous['AMT_APPLICATION'].replace(0,np.nan, inplace = True)
previous['AMT_CREDIT'].replace(0,np.nan, inplace = True)
previous['AMT_GOODS_PRICE'].replace(0,np.nan,inplace =True)
previous['RATE_DOWN_PAYMENT'].replace(0, np.nan, inplace = True)
previous['AMT_ANNUITY'].replace(0, np.nan, inplace = True)
previous['CNT_PAYMENT'].replace(0, np.nan, inplace = True)
for i in ['Revolving loans','Cash loans', 'Consumer loans']:
    tmp = previous[(previous['NAME_CONTRACT_TYPE'] == i) & (previous['DAYS_LAST_DUE'] == 365243)]

    for df in [train,test]:
        tmp1 = tmp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
        tmp1.columns = ['SK_ID_CURR','des']
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
        df['count_notfinish_' + "_".join(i.lower().split())] = tmp_merge['des'].fillna(0)
for i in ['Revolving loans','Cash loans', 'Consumer loans']:
    tmp = previous[(previous['NAME_CONTRACT_TYPE'] == i) & (previous['DAYS_LAST_DUE'] == 365243)]

    for df in [train,test]:
        tmp1 = tmp.groupby(['SK_ID_CURR'])['AMT_CREDIT'].agg({"returns": [np.mean, np.sum]})\
                                                                .reset_index()
        tmp1.columns = ['SK_ID_CURR','des1','des2']
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
        df['mean_notfinish_' + "_".join(i.lower().split())] = tmp_merge['des1'].fillna(0)
        df['sum_notfinish_' + "_".join(i.lower().split())] = tmp_merge['des2'].fillna(0)
for i in ['Revolving loans','Cash loans', 'Consumer loans']:
    tmp = previous[(previous['NAME_CONTRACT_TYPE'] == i) & (previous['DAYS_TERMINATION'] == 365243)]

    for df in [train,test]:
        tmp1 = tmp.groupby(['SK_ID_CURR'])['AMT_ANNUITY'].agg({"returns": [np.mean, np.sum]})\
                                                                .reset_index()
        tmp1.columns = ['SK_ID_CURR','des1','des2']
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
        df['mean_annuity_notfinish_' + "_".join(i.lower().split())] = tmp_merge['des1'].fillna(0)
        df['sum_annuity_notfinish_' + "_".join(i.lower().split())] = tmp_merge['des2'].fillna(0)

previous['SELLERPLACE_AREA'].replace(0, np.nan, inplace = True)
previous['SELLERPLACE_AREA'].replace(-1, np.nan, inplace = True)
previous['DAYS_TERMINATION'].replace(365243, np.nan, inplace = True)
previous['DAYS_LAST_DUE'].replace(365243, np.nan, inplace = True)
previous['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace = True)
previous['sooner'] = (previous['DAYS_LAST_DUE_1ST_VERSION'] - previous['DAYS_LAST_DUE'])/(previous['DAYS_LAST_DUE_1ST_VERSION']-previous['DAYS_DECISION'])
previous['duration'] = previous['DAYS_TERMINATION'] - previous['DAYS_DECISION']
previous['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace = True)
train['DAYS_EMPLOYED'] = train['DAYS_EMPLOYED'].replace(365243, np.nan)
test['DAYS_EMPLOYED'] = test['DAYS_EMPLOYED'].replace(365243, np.nan)


tmp = train[train['DAYS_LAST_PHONE_CHANGE'] >= 0].index
train['DAYS_LAST_PHONE_CHANGE'].iloc[tmp] = np.nan
tmp = test[test['DAYS_LAST_PHONE_CHANGE'] >= 0].index
test['DAYS_LAST_PHONE_CHANGE'].iloc[tmp] = np.nan

for df in [train, test]:
    df['ORGANIZATION_TYPE_v2'] = df['ORGANIZATION_TYPE']
    for i in range(1,4):
        df['ORGANIZATION_TYPE_v2'].replace('Business Entity Type ' + str(i), 'Business', inplace = True)
    for i in range(1,14):
        df['ORGANIZATION_TYPE_v2'].replace('Industry: type ' + str(i), 'Industry', inplace = True)
    for i in range(1,8):
        df['ORGANIZATION_TYPE_v2'].replace('Trade: type ' + str(i), 'Trade', inplace = True)
    for i in range(1,8):
        df['ORGANIZATION_TYPE_v2'].replace('Transport: type ' + str(i), 'Transport', inplace = True)
    df['ORGANIZATION_TYPE_v2'].replace('Other','XNA', inplace = True)
train.head()
tmp = previous[(previous['NAME_CONTRACT_STATUS'] == 'Approved') &(previous['NAME_CONTRACT_TYPE'].isin(['Consumer loans','Cash loans', 'Revolving loans']))].groupby(['SK_ID_CURR'])['AMT_CREDIT']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_credit_master'] = tmp_merge['des1']
    df['max_amt_credit_master'] = tmp_merge['des2']
    df['mean_amt_credit_master'] = tmp_merge['des3']
tmp = previous[(previous['NAME_CONTRACT_STATUS'] == 'Approved') &(previous['NAME_CONTRACT_TYPE'].isin(['Cash loans']))].groupby(['SK_ID_CURR'])['AMT_APPLICATION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_app'] = tmp_merge['des1']
    df['max_amt_app'] = tmp_merge['des2']
    df['mean_amt_app'] = tmp_merge['des3']

tmp = previous[(previous['NAME_CONTRACT_STATUS'] == 'Approved') &(previous['NAME_CONTRACT_TYPE'].isin(['Consumer loans']))].groupby(['SK_ID_CURR'])['AMT_APPLICATION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_app_v1'] = tmp_merge['des1']
    df['max_amt_app_v1'] = tmp_merge['des2']
    df['mean_amt_app_v1'] = tmp_merge['des3']

tmp = previous[(previous['NAME_CONTRACT_STATUS']  == 'Approved') &(previous['NAME_CONTRACT_TYPE'].isin(['Revolving loans']))].groupby(['SK_ID_CURR'])['AMT_CREDIT']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_card'] = tmp_merge['des1']
    df['max_amt_card'] = tmp_merge['des2']
    df['mean_amt_card'] = tmp_merge['des3']
tmp = previous[(previous['NAME_CONTRACT_STATUS'].isin(['Refused','Canceled'])) &(previous['NAME_CONTRACT_TYPE'].isin(['Cash loans']))].groupby(['SK_ID_CURR'])['AMT_APPLICATION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_app_fail'] = tmp_merge['des1']
    df['max_amt_app_fail'] = tmp_merge['des2']
    df['mean_amt_app_fail'] = tmp_merge['des3']

tmp = previous[(previous['NAME_CONTRACT_STATUS'].isin(['Refused','Canceled']))  &(previous['NAME_CONTRACT_TYPE'].isin(['Consumer loans']))].groupby(['SK_ID_CURR'])['AMT_APPLICATION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_app_v1_fail'] = tmp_merge['des1']
    df['max_amt_app_v1_fail'] = tmp_merge['des2']
    df['mean_amt_app_v1_fail'] = tmp_merge['des3']

tmp = previous[(previous['NAME_CONTRACT_STATUS'].isin(['Refused','Canceled']))  &(previous['NAME_CONTRACT_TYPE'].isin(['Revolving loans']))].groupby(['SK_ID_CURR'])['AMT_CREDIT']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_card_fail'] = tmp_merge['des1']
    df['max_amt_card_fail'] = tmp_merge['des2']
    df['mean_amt_card_fail'] = tmp_merge['des3']
tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Cash loans']))].groupby(['SK_ID_CURR'])['RATE_DOWN_PAYMENT']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_goods'] = tmp_merge['des1']
    df['max_amt_goods'] = tmp_merge['des2']
    df['mean_amt_goods'] = tmp_merge['des3']

tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Consumer loans']))].groupby(['SK_ID_CURR'])['RATE_DOWN_PAYMENT']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_goods_v1'] = tmp_merge['des1']
    df['max_amt_goods_v1'] = tmp_merge['des2']
    df['mean_amt_goods_v1'] = tmp_merge['des3']


tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Cash loans','Consumer loans']))].groupby(['SK_ID_CURR'])['AMT_ANNUITY']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_annuity'] = tmp_merge['des1']
    df['max_amt_annuity'] = tmp_merge['des2']
    df['mean_amt_annuity'] = tmp_merge['des3']

#tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Consumer loans']))].groupby(['SK_ID_CURR'])['AMT_ANNUITY']\
#                                                                .agg({"returns": [np.min, np.max,np.mean]})\
#                                                                .reset_index()
#tmp.columns = ['SK_ID_CURR','des1','des2','des3']
#for df in [train,test]:
#    tmp_merge = df[['SK_ID_CURR']]
#    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
#    df['min_amt_annuity_v1'] = tmp_merge['des1']
#    df['max_amt_annuity_v1'] = tmp_merge['des2']
#    df['mean_amt_annuity_v1'] = tmp_merge['des3']

tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Revolving loans']))].groupby(['SK_ID_CURR'])['AMT_ANNUITY']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_amt_card_annuity'] = tmp_merge['des1']
    df['max_amt_card_annuity'] = tmp_merge['des2']
    df['mean_amt_card_annuity'] = tmp_merge['des3']
tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Cash loans','Consumer loans']))].groupby(['SK_ID_CURR'])['CNT_PAYMENT']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_cntpay'] = tmp_merge['des1']
    df['max_cntpay'] = tmp_merge['des2']
    df['mean_cntpay'] = tmp_merge['des3']

tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Consumer loans']))].groupby(['SK_ID_CURR'])['CNT_PAYMENT']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_cntpay_v1'] = tmp_merge['des1']
    df['max_cntpay_v1'] = tmp_merge['des2']
    df['mean_cntpay_v1'] = tmp_merge['des3']


tmp = previous[(previous['NAME_CONTRACT_STATUS'] == 'Approved') & (previous['AMT_APPLICATION'] > 0) & (previous['NAME_CONTRACT_TYPE'].isin(['Cash loans','Consumer loans']))].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','AMT_APPLICATION']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_app'] = tmp_merge['des']

tmp = previous[(previous['NAME_CONTRACT_STATUS'] == 'Approved') &(previous['AMT_APPLICATION'] > 0) & (previous['NAME_CONTRACT_TYPE'].isin(['Cash loans','Consumer loans']))].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','AMT_CREDIT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_credit'] = tmp_merge['des']

tmp = previous[(previous['NAME_CONTRACT_STATUS'] == 'Approved') &(previous['AMT_CREDIT'] > 0) & (previous['NAME_CONTRACT_TYPE'].isin(['Revolving loans']))].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','AMT_CREDIT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_card'] = tmp_merge['des']
tmp = previous[(previous['NAME_CONTRACT_STATUS'] != 'Approved') & (previous['AMT_APPLICATION'] > 0) & (previous['NAME_CONTRACT_TYPE'].isin(['Cash loans','Consumer loans']))].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','AMT_APPLICATION']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_app_fail'] = tmp_merge['des']

tmp = previous[(previous['NAME_CONTRACT_STATUS'] != 'Approved') &(previous['AMT_APPLICATION'] > 0) & (previous['NAME_CONTRACT_TYPE'].isin(['Cash loans','Consumer loans']))].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','AMT_CREDIT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_credit_fail'] = tmp_merge['des']

tmp = previous[(previous['NAME_CONTRACT_STATUS'] != 'Approved') &(previous['AMT_CREDIT'] > 0) & (previous['NAME_CONTRACT_TYPE'].isin(['Revolving loans']))].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','AMT_CREDIT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_card_fail'] = tmp_merge['des']
tmp = previous[(previous['NAME_CONTRACT_STATUS'] == 'Approved') & (previous['RATE_DOWN_PAYMENT'] > 0)].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','RATE_DOWN_PAYMENT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_ratedown'] = tmp_merge['des']
    
tmp = previous[(previous['NAME_CONTRACT_STATUS'] != 'Approved') & (previous['RATE_DOWN_PAYMENT'] > 0)].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','RATE_DOWN_PAYMENT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_ratedown_fail'] = tmp_merge['des']
tmp = previous[(previous['NAME_CONTRACT_TYPE'].isin(['Consumer loans','Cash loans'])) & (previous['CNT_PAYMENT'] > 0)].sort_values(by=['SK_ID_CURR','DAYS_DECISION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-1).reset_index()
tmp = tmp[['SK_ID_CURR','CNT_PAYMENT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_recent_cntpay'] = tmp_merge['des']
    

tmp = previous[previous['AMT_CREDIT'] > 0]
for i in ['Cash loans','Consumer loans','Revolving loans']:
    for df in [train,test]:
        tmp1 = tmp[tmp['NAME_CONTRACT_TYPE'] == i].groupby(['SK_ID_CURR'])['AMT_CREDIT'].count().reset_index()
        tmp1.columns = ['SK_ID_CURR','des']
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
        df['count_' + "_".join(i.lower().split())] = tmp_merge['des']
tmp = previous[previous['AMT_CREDIT'].isnull()]
for i in ['Cash loans','Consumer loans','Revolving loans']:
    for df in [train,test]:
        tmp1 = tmp[tmp['NAME_CONTRACT_TYPE'] == i].groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
        tmp1.columns = ['SK_ID_CURR','des']
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
        df['count_null_' + "_".join(i.lower().split())] = tmp_merge['des']
tmp = previous[previous['AMT_CREDIT'] > 0].groupby(['SK_ID_CURR'])['DAYS_DECISION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_day_decision'] = tmp_merge['des1']
    df['max_day_decision'] = tmp_merge['des2']
    df['mean_day_decision'] = tmp_merge['des3']
tmp = previous[previous['NAME_CONTRACT_STATUS'] != 'Approved'].groupby(['SK_ID_CURR'])['DAYS_DECISION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_day_decision_fail'] = tmp_merge['des1']
    df['max_day_decision_fail'] = tmp_merge['des2']
    df['mean_day_decision_fail'] = tmp_merge['des3']
tmp = previous[(~previous['NAME_CASH_LOAN_PURPOSE'].isin(['XAP','XNA']))]

for df in [train,test]:
    tmp1 = tmp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
    tmp1.columns = ['SK_ID_CURR','des']
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
    df['count_clear_reason'] = tmp_merge['des']
tmp = previous.groupby(['SK_ID_CURR'])['DAYS_TERMINATION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_day_termination'] = tmp_merge['des1']
    df['max_day_termination'] = tmp_merge['des2']
    df['mean_day_termination'] = tmp_merge['des3']
tmp = previous.groupby(['SK_ID_CURR'])['DAYS_LAST_DUE_1ST_VERSION']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_day_lastdue'] = tmp_merge['des1']
    df['max_day_lastdue'] = tmp_merge['des2']
    df['mean_day_lastdue'] = tmp_merge['des3']
tmp = previous[~previous['DAYS_LAST_DUE_1ST_VERSION'].isnull()].sort_values(by=['SK_ID_CURR','DAYS_LAST_DUE_1ST_VERSION'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-2).reset_index()
tmp = tmp[['SK_ID_CURR','DAYS_LAST_DUE_1ST_VERSION']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['2nd_day_lastdue'] = tmp_merge['des']
tmp = previous.groupby(['SK_ID_CURR'])['sooner']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_sooner'] = tmp_merge['des1']
    df['max_sooner'] = tmp_merge['des2']
    df['mean_sooner'] = tmp_merge['des3']
tmp = previous.groupby(['SK_ID_CURR'])['SELLERPLACE_AREA']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_seller'] = tmp_merge['des1']
    df['max_seller'] = tmp_merge['des2']
    df['mean_seller'] = tmp_merge['des3']
print(train.shape, test.shape)
for i in ['middle','low_normal','high','low_action']:
    tmp = previous[(previous['NAME_CONTRACT_TYPE'] == 'Cash loans') & (previous['NAME_YIELD_GROUP'] == i)]

    for df in [train,test]:
        tmp1 = tmp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
        tmp1.columns = ['SK_ID_CURR','des']
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
        df['count_' + str(i)] = tmp_merge['des'].fillna(0)
for df in [train,test]:
    df['tmp'] = df[['count_middle','count_low_normal','count_high','count_low_action']].sum(axis=1)
    for i in ['middle','low_normal','high','low_action']:
        df['ratio_' + i] = df['count_' + i]/df['tmp']
        #df['ratio_' + i] = df['ratio_' + i].fillna(0)
for i in ['middle','low_normal','high','low_action']:
    tmp = previous[(previous['NAME_CONTRACT_TYPE'] == 'Consumer loans') & (previous['NAME_YIELD_GROUP'] == i)]

    for df in [train,test]:
        tmp1 = tmp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
        tmp1.columns = ['SK_ID_CURR','des']
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp1, on=['SK_ID_CURR'], how='left')
        df['count_' + str(i) + '_v1'] = tmp_merge['des'].fillna(0)
for df in [train,test]:
    df['tmp'] = df[['count_middle_v1','count_low_normal_v1','count_high_v1','count_low_action_v1']].sum(axis=1)
    for i in ['middle','low_normal','high','low_action']:
        df['ratio_' + i +"_v1"] = df['count_' + i + "_v1"]/df['tmp']
        #df['ratio_' + i] = df['ratio_' + i].fillna(0)
previous['tmp'] = (previous['AMT_ANNUITY'] * previous['CNT_PAYMENT'])/previous['AMT_CREDIT']
tmp = previous[(previous['NAME_CONTRACT_TYPE'] == 'Cash loans') & (previous['NAME_CONTRACT_STATUS'] != 'Approved')].groupby(['SK_ID_CURR'])['tmp']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_interest'] = tmp_merge['des1']
    df['max_interest'] = tmp_merge['des2']
    df['mean_interest'] = tmp_merge['des3']
    
tmp = previous[(previous['NAME_CONTRACT_TYPE'] == 'Consumer loans') & (previous['NAME_CONTRACT_STATUS'] != 'Approved')].groupby(['SK_ID_CURR'])['tmp']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_interest_v1'] = tmp_merge['des1']
    df['max_interest_v1'] = tmp_merge['des2']
    df['mean_interest_v1'] = tmp_merge['des3']
tmp = previous.groupby(['SK_ID_CURR'])['DAYS_FIRST_DRAWING']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_firstdraw'] = tmp_merge['des1']
    df['max_firstdraw'] = tmp_merge['des2']
    df['mean_firstdraw'] = tmp_merge['des3']
previous['tmp'] = previous['DAYS_FIRST_DRAWING'] - previous['DAYS_DECISION']
tmp = previous.groupby(['SK_ID_CURR'])['tmp']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_firstdraw_decision'] = tmp_merge['des1']
    df['max_firstdraw_decision'] = tmp_merge['des2']
    df['mean_firstdraw_decision'] = tmp_merge['des3']
previous['tmp'] = previous['DAYS_FIRST_DUE'] - previous['DAYS_FIRST_DRAWING']
tmp = previous.groupby(['SK_ID_CURR'])['tmp']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_firstdraw_firstdue'] = tmp_merge['des1']
    df['max_firstdraw_firstdue'] = tmp_merge['des2']
    df['mean_firstdraw_firstdue'] = tmp_merge['des3']
previous['tmp'] = previous['DAYS_LAST_DUE'] - previous['DAYS_FIRST_DRAWING']
tmp = previous.groupby(['SK_ID_CURR'])['tmp']\
                                                                .agg({"returns": [np.min, np.max,np.mean]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_firstdraw_lastdue'] = tmp_merge['des1']
    df['max_firstdraw_lastdue'] = tmp_merge['des2']
    df['mean_firstdraw_lastdue'] = tmp_merge['des3']
tmp = bureau[(bureau['CREDIT_ACTIVE'] == "Active") ].groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['count_active_bureau'] = tmp_merge['des'].fillna(0)

tmp = bureau[bureau['CREDIT_ACTIVE'] != "Active"].groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['count_closed_bureau'] = tmp_merge['des'].fillna(0)
tmp = bureau[(bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['count_active_bureau_v2'] = tmp_merge['des'].fillna(0)

tmp = bureau[(bureau['CREDIT_ACTIVE'] != "Active")& (bureau['CREDIT_TYPE'] == "Credit card")].groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['count_closed_bureau_v2'] = tmp_merge['des'].fillna(0)
tmp = bureau[(~bureau['CREDIT_TYPE'].isin(["Credit card","Consumer credit"]))].groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['count_active_bureau_v3'] = tmp_merge['des'].fillna(0)

tmp = bureau[(bureau['CREDIT_ACTIVE'] != "Active")& (~bureau['CREDIT_TYPE'].isin(["Credit card","Consumer credit"]))].groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['count_closed_bureau_v3'] = tmp_merge['des'].fillna(0)
bureau['AMT_CREDIT_SUM'].replace(0, np.nan, inplace = True)
tmp = bureau[(bureau['CREDIT_ACTIVE'] == "Active") & (bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_active_credit_bureau'] = tmp_merge['des1']
    df['max_active_credit_bureau'] = tmp_merge['des2']
    df['mean_active_credit_bureau'] = tmp_merge['des3']
    df['sum_active_credit_bureau'] = tmp_merge['des4']
    
tmp = bureau[(bureau['CREDIT_ACTIVE'] != "Active") & (bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_closed_credit_bureau'] = tmp_merge['des1']
    df['max_closed_credit_bureau'] = tmp_merge['des2']
    df['mean_closed_credit_bureau'] = tmp_merge['des3']
    df['sum_closed_credit_bureau'] = tmp_merge['des4']
bureau['AMT_CREDIT_SUM'].replace(0, np.nan, inplace = True)
tmp = bureau[(bureau['CREDIT_TYPE'] == "Credit card")].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_active_credit_bureau_v1'] = tmp_merge['des1']
    df['max_active_credit_bureau_v1'] = tmp_merge['des2']
    df['mean_active_credit_bureau_v1'] = tmp_merge['des3']
    df['sum_active_credit_bureau_v1'] = tmp_merge['des4']
    
bureau['AMT_CREDIT_SUM'].replace(0, np.nan, inplace = True)
tmp = bureau[(bureau['CREDIT_TYPE'] == "Car loan")].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_active_credit_bureau_v2'] = tmp_merge['des1']
    df['max_active_credit_bureau_v2'] = tmp_merge['des2']
    df['mean_active_credit_bureau_v2'] = tmp_merge['des3']
    df['sum_active_credit_bureau_v2'] = tmp_merge['des4']
    
bureau['AMT_CREDIT_SUM'].replace(0, np.nan, inplace = True)
tmp = bureau[ (~bureau['CREDIT_TYPE'].isin(["Credit card","Consumer credit","Car loan"]))].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_active_credit_bureau_v3'] = tmp_merge['des1']
    df['max_active_credit_bureau_v3'] = tmp_merge['des2']
    df['mean_active_credit_bureau_v3'] = tmp_merge['des3']
    df['sum_active_credit_bureau_v3'] = tmp_merge['des4']
    
tmp = bureau.groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_credit_bureau'] = tmp_merge['des1']
    df['max_credit_bureau'] = tmp_merge['des2']
    df['mean_credit_bureau'] = tmp_merge['des3']
    df['sum_credit_bureau'] = tmp_merge['des4']
tmp = bureau[(bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_endate_bureau'] = tmp_merge['des1']
    df['max_endate_bureau'] = tmp_merge['des2']
    df['mean_endate_bureau'] = tmp_merge['des3']
    df['sum_endate_bureau'] = tmp_merge['des4']

tmp = bureau[(~bureau['DAYS_CREDIT_ENDDATE'].isnull()) & ((bureau['CREDIT_TYPE'] == "Consumer credit"))].sort_values(by=['SK_ID_CURR','DAYS_CREDIT_ENDDATE'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-2).reset_index()
tmp = tmp[['SK_ID_CURR','DAYS_CREDIT_ENDDATE']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_endate_bureau'] = tmp_merge['des']
tmp = bureau[(bureau['CREDIT_TYPE'] == "Car loan")].groupby(['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_endate_bureau_v1'] = tmp_merge['des1']
    df['max_endate_bureau_v1'] = tmp_merge['des2']
    df['mean_endate_bureau_v1'] = tmp_merge['des3']
    df['sum_endate_bureau_v1'] = tmp_merge['des4']

tmp = bureau[(~bureau['DAYS_CREDIT_ENDDATE'].isnull()) & ((bureau['CREDIT_TYPE'] == "Car loan"))].sort_values(by=['SK_ID_CURR','DAYS_CREDIT_ENDDATE'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-2).reset_index()
tmp = tmp[['SK_ID_CURR','DAYS_CREDIT_ENDDATE']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_endate_bureau_v1'] = tmp_merge['des']
tmp = bureau[(bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['DAYS_CREDIT'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_startdate_bureau'] = tmp_merge['des1']
    df['max_startdate_bureau'] = tmp_merge['des2']
    df['mean_startdate_bureau'] = tmp_merge['des3']
    df['sum_startdate_bureau'] = tmp_merge['des4']
tmp = bureau[(~bureau['DAYS_CREDIT_ENDDATE'].isnull()) & ((bureau['CREDIT_TYPE'] == "Consumer credit"))].sort_values(by=['SK_ID_CURR','DAYS_CREDIT_ENDDATE'])
tmp = tmp.groupby(['SK_ID_CURR']).nth(-2).reset_index()
tmp = tmp[['SK_ID_CURR','DAYS_ENDDATE_FACT']]
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['1st_endatefact_bureau'] = tmp_merge['des']
tmp = bureau[(bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['DAYS_ENDDATE_FACT'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_endatefact_bureau'] = tmp_merge['des1']
    df['max_endatefact_bureau'] = tmp_merge['des2']
    df['mean_endatefact_bureau'] = tmp_merge['des3']
    df['sum_endatefact_bureau'] = tmp_merge['des4']
bureau['tmp'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
tmp = bureau.groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_deltaendate_bureau'] = tmp_merge['des1']
    df['max_deltaendate_bureau'] = tmp_merge['des2']
    df['mean_deltaendate_bureau'] = tmp_merge['des3']
    df['sum_deltaendate_bureau'] = tmp_merge['des4']
bureau['tmp'] = (bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT'])
tmp = bureau[(bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_duration_bureau'] = tmp_merge['des1']
    df['max_duration_bureau'] = tmp_merge['des2']
    df['mean_duration_bureau'] = tmp_merge['des3']
    df['sum_duration_bureau'] = tmp_merge['des4']
    
bureau['tmp'] = (bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT'])
tmp = bureau[(bureau['CREDIT_TYPE'] != "Consumer credit")].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_duration_bureau_v1'] = tmp_merge['des1']
    df['max_duration_bureau_v1'] = tmp_merge['des2']
    df['mean_duration_bureau_v1'] = tmp_merge['des3']
    df['sum_duration_bureau_v1'] = tmp_merge['des4']
bureau['tmp'] = (bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT'])
tmp = bureau[(bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_durationfact_bureau'] = tmp_merge['des1']
    df['max_durationfact_bureau'] = tmp_merge['des2']
    df['mean_durationfact_bureau'] = tmp_merge['des3']
    df['sum_durationfact_bureau'] = tmp_merge['des4']
bureau['tmp'] = (bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE'])/(bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE'])
tmp = bureau[ (bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_sooner_bureau'] = tmp_merge['des1']
    df['max_sooner_bureau'] = tmp_merge['des2']
    df['mean_sooner_bureau'] = tmp_merge['des3']
    df['sum_sooner_bureau'] = tmp_merge['des4']
bureau['tmp'] = (bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE'])/(bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE'])
tmp = bureau[ (~bureau['CREDIT_TYPE'].isin(['Credit card','Consumer credit']))].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_sooner_bureau_v1'] = tmp_merge['des1']
    df['max_sooner_bureau_v1'] = tmp_merge['des2']
    df['mean_sooner_bureau_v1'] = tmp_merge['des3']
    df['sum_sooner_bureau_v1'] = tmp_merge['des4']
bureau['tmp'] = (bureau['AMT_CREDIT_SUM'])/(bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE'])
tmp = bureau[bureau['CREDIT_TYPE'] == "Credit card"].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_annuity_bureau'] = tmp_merge['des1']
    df['max_annuity_bureau'] = tmp_merge['des2']
    df['mean_annuity_bureau'] = tmp_merge['des3']
    df['sum_annuity_bureau'] = tmp_merge['des4']
bureau['tmp'] = (bureau['AMT_CREDIT_SUM'])/(bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE'])
tmp = bureau[(bureau['CREDIT_ACTIVE'] == "Active") & (bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_annuity_bureau_v1'] = tmp_merge['des1']
    df['max_annuity_bureau_v1'] = tmp_merge['des2']
    df['mean_annuity_bureau_v1'] = tmp_merge['des3']
    df['sum_annuity_bureau_v1'] = tmp_merge['des4']
    
tmp = bureau[(bureau['CREDIT_ACTIVE'] == "Active") & (bureau['CREDIT_TYPE'] == "Consumer credit")].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_annuity_bureau_v2'] = tmp_merge['des1']
    df['max_annuity_bureau_v2'] = tmp_merge['des2']
    df['mean_annuity_bureau_v2'] = tmp_merge['des3']
    df['sum_annuity_bureau_v2'] = tmp_merge['des4']
bureau['tmp'] = (bureau['AMT_CREDIT_SUM'])/(bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE'])
tmp = bureau[~bureau['CREDIT_TYPE'].isin(["Credit card","Consumer credit"])].groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_annuity_bureau_v3'] = tmp_merge['des1']
    df['max_annuity_bureau_v3'] = tmp_merge['des2']
    df['mean_annuity_bureau_v3'] = tmp_merge['des3']
    df['sum_annuity_bureau_v3'] = tmp_merge['des4']
bureau['AMT_CREDIT_SUM_DEBT_v1'] = bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan)
tmp = bureau[(bureau['CREDIT_ACTIVE'] == "Active") & (bureau['CREDIT_TYPE'].isin(["Credit card"]))].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT_v1'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_debt_bureau'] = tmp_merge['des1']
    df['max_debt_bureau'] = tmp_merge['des2']
    df['mean_debt_bureau'] = tmp_merge['des3']
    df['sum_debt_bureau'] = tmp_merge['des4']
    
tmp = bureau[(bureau['CREDIT_ACTIVE'] == "Active") & (bureau['CREDIT_TYPE'].isin(["Consumer credit"]))].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT_v1'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_debt_bureau_v1'] = tmp_merge['des1']
    df['max_debt_bureau_v1'] = tmp_merge['des2']
    df['mean_debt_bureau_v1'] = tmp_merge['des3']
    df['sum_debt_bureau_v1'] = tmp_merge['des4']
bureau['AMT_CREDIT_SUM_LIMIT_v1'] = bureau['AMT_CREDIT_SUM_LIMIT'].replace(0, np.nan)
tmp = bureau[ (bureau['CREDIT_TYPE'].isin(["Credit card"]))].groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM_LIMIT_v1'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_limit_bureau'] = tmp_merge['des1']
    df['max_limit_bureau'] = tmp_merge['des2']
    df['mean_limit_bureau'] = tmp_merge['des3']
    df['sum_limit_bureau'] = tmp_merge['des4']

bureau['AMT_CREDIT_MAX_OVERDUE_v1'] = bureau['AMT_CREDIT_MAX_OVERDUE'].replace(0,np.nan)
tmp = bureau[(bureau['CREDIT_TYPE'].isin(["Consumer credit"]))].groupby(['SK_ID_CURR'])['AMT_CREDIT_MAX_OVERDUE_v1'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_overdue_bureau'] = tmp_merge['des1'].fillna(0)
    df['max_overdue_bureau'] = tmp_merge['des2'].fillna(0)
    df['mean_overdue_bureau'] = tmp_merge['des3'].fillna(0)
    df['sum_overdue_bureau'] = tmp_merge['des4'].fillna(0)
bureau['tmp'] = bureau['AMT_CREDIT_SUM_DEBT']/bureau['AMT_CREDIT_SUM']
tmp = bureau.groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_ratio_debt_credit_bureau'] = tmp_merge['des1']
    df['max_ratio_debt_credit_bureau'] = tmp_merge['des2']
    df['mean_ratio_debt_credit_bureau'] = tmp_merge['des3']
    df['sum_ratio_debt_credit_bureau'] = tmp_merge['des4']
bureau['tmp'] = bureau['AMT_ANNUITY'].fillna(0)
tmp = bureau.groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_annuity_bureau_v2'] = tmp_merge['des1']
    df['max_annuity_bureau_v2'] = tmp_merge['des2']
    df['mean_annuity_bureau_v2'] = tmp_merge['des3']
    df['sum_annuity_bureau_v2'] = tmp_merge['des4']
install = pd.read_csv("../input/installments_payments.csv")
tmp = install.groupby(['SK_ID_PREV','SK_ID_CURR']).agg({'NUM_INSTALMENT_NUMBER':["count","max"]}).reset_index()
tmp.columns = ['SK_ID_PREV','SK_ID_CURR','count','max']
tmp['delta'] = tmp['count'] - tmp['max']
tmp = tmp.merge(previous[['SK_ID_PREV','CNT_PAYMENT','DAYS_LAST_DUE','NAME_CONTRACT_TYPE']], on=['SK_ID_PREV'], how='left')

tmp_1 = tmp.groupby(['SK_ID_CURR'])['delta'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp_1.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp_1, on=['SK_ID_CURR'], how='left')
    df['min_delta_num_install'] = tmp_merge['des1']
    df['max_delta_num_install'] = tmp_merge['des2']
    df['mean_delta_num_install'] = tmp_merge['des3']
    df['sum_delta_num_install'] = tmp_merge['des4']
tmp = install.groupby(['SK_ID_PREV','SK_ID_CURR']).agg({'NUM_INSTALMENT_VERSION':["count","max"]}).reset_index()
tmp.columns = ['SK_ID_PREV','SK_ID_CURR','count','max']
#tmp['delta'] = tmp['count'] - tmp['max']
tmp = tmp.merge(previous[['SK_ID_PREV','CNT_PAYMENT','DAYS_LAST_DUE','NAME_CONTRACT_TYPE']], on=['SK_ID_PREV'], how='left')

tmp_1 = tmp.groupby(['SK_ID_CURR'])['max'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp_1.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp_1, on=['SK_ID_CURR'], how='left')
    df['min_max_version_install'] = tmp_merge['des1']
    df['max_max_version_install'] = tmp_merge['des2']
    df['mean_max_version_install'] = tmp_merge['des3']
    df['sum_max_version_install'] = tmp_merge['des4']
tmp = install.groupby(['SK_ID_PREV','SK_ID_CURR']).agg({'NUM_INSTALMENT_NUMBER':["count","max"]}).reset_index()
tmp.columns = ['SK_ID_PREV','SK_ID_CURR','count','max']
tmp['delta'] = tmp['count']/tmp['max']
tmp = tmp.merge(previous[['SK_ID_PREV','CNT_PAYMENT','DAYS_LAST_DUE','NAME_CONTRACT_TYPE']], on=['SK_ID_PREV'], how='left')

tmp_1 = tmp.groupby(['SK_ID_CURR'])['delta'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp_1.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp_1, on=['SK_ID_CURR'], how='left')
    df['min_ratio_num_install'] = tmp_merge['des1']
    df['max_ratio_num_install'] = tmp_merge['des2']
    df['mean_ratio_num_install'] = tmp_merge['des3']
    df['sum_ratio_num_install'] = tmp_merge['des4']
tmp = install[['SK_ID_PREV','SK_ID_CURR','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT']].drop_duplicates()
tmp = tmp.groupby(['SK_ID_PREV','SK_ID_CURR'])['AMT_INSTALMENT'].sum().reset_index()
tmp.columns = ['SK_ID_PREV','SK_ID_CURR','need_to_pay']
tmp_1 = install.groupby(['SK_ID_PREV'])['AMT_PAYMENT'].sum().reset_index()
tmp_1.columns = ['SK_ID_PREV','paid']
tmp = tmp.merge(tmp_1, on=['SK_ID_PREV'], how='left')
payment_history = tmp

payment_history['ratio'] = payment_history['paid']/payment_history['need_to_pay']
payment_history['delta'] = payment_history['need_to_pay'] - payment_history['paid']

payment_history = payment_history.merge(previous[['SK_ID_PREV','AMT_ANNUITY','CNT_PAYMENT','NAME_CONTRACT_TYPE']], \
                                       on = ['SK_ID_PREV'], how='left')

payment_history['all_credit'] = payment_history['AMT_ANNUITY'] * payment_history['CNT_PAYMENT']

payment_history['ratio'] = payment_history['paid']/payment_history['all_credit']
payment_history['delta'] = payment_history['all_credit'] - payment_history['paid']

tmp = install.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].mean().reset_index()
tmp.columns = ['SK_ID_PREV','mean_version']
payment_history = payment_history.merge(tmp, on=['SK_ID_PREV'], how='left')

tmp = payment_history[payment_history['mean_version'] > 0].groupby(['SK_ID_CURR'])['ratio'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_ratio_paid_install'] = tmp_merge['des1']
    df['max_ratio_paid_install'] = tmp_merge['des2']
    df['mean_ratio_paid_install'] = tmp_merge['des3']
    df['sum_ratio_paid_install'] = tmp_merge['des4']

tmp = payment_history[payment_history['mean_version'] > 0].groupby(['SK_ID_CURR'])['delta'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_delta_paid_install'] = tmp_merge['des1']
    df['max_delta_paid_install'] = tmp_merge['des2']
    df['mean_delta_paid_install'] = tmp_merge['des3']
    df['sum_delta_paid_install'] = tmp_merge['des4']
tmp = install.groupby(['SK_ID_PREV','SK_ID_CURR']).agg({'NUM_INSTALMENT_NUMBER':["count","max"]}).reset_index()
tmp.columns = ['SK_ID_PREV','SK_ID_CURR','count','max']
tmp['delta'] = tmp['count']/tmp['max']
tmp = tmp.merge(previous[['SK_ID_PREV','CNT_PAYMENT','DAYS_LAST_DUE','NAME_CONTRACT_TYPE']], on=['SK_ID_PREV'], how='left')

tmp_1 = tmp.groupby(['SK_ID_CURR'])['max'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp_1.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp_1, on=['SK_ID_CURR'], how='left')
    df['min_max_num_install'] = tmp_merge['des1']
    df['max_max_num_install'] = tmp_merge['des2']
    df['mean_max_num_install'] = tmp_merge['des3']
    df['sum_max_num_install'] = tmp_merge['des4']
tmp = install.groupby(['SK_ID_PREV'])['AMT_PAYMENT'].max().reset_index()
tmp.columns = ['SK_ID_PREV','max_install']
payment_history = payment_history.merge(tmp, on=['SK_ID_PREV'], how='left')
payment_history['tmp'] = payment_history['max_install']/payment_history['AMT_ANNUITY']
tmp = payment_history[payment_history['mean_version'] > 0].groupby(['SK_ID_CURR'])['tmp']
tmp = install.groupby(['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
tmp.columns = ['SK_ID_PREV','max_num_install']
payment_history = payment_history.merge(tmp, on=['SK_ID_PREV'], how='left')
tmp = install[install['AMT_INSTALMENT'] > install['AMT_PAYMENT']]

tmp = tmp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
tmp.columns = ['SK_ID_CURR','des']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['count_small_payment'] = tmp_merge['des'].fillna(0)
    
for i in [0, 5, 10, 15, 20, 25, 30,  40,  50, 60]:
    print(i)

    tmp = install[(install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']) > i]

    tmp = tmp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
    tmp.columns = ['SK_ID_CURR','des']
    for df in [train,test]:
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
        df['count_late_payment_' + str(i)] = tmp_merge['des'].fillna(0)

install['tmp'] = install['AMT_PAYMENT']/install['AMT_INSTALMENT']
#tmp = install#[(install['tmp'] > 0.2) & (install['tmp'] < 0.8)]
for i in range(10):
    print(i)

    tmp = install[(install['tmp'] > i/10 )& (install['tmp'] < ((i+1)/10))]

    tmp = tmp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index()
    tmp.columns = ['SK_ID_CURR','des']
    for df in [train,test]:
        tmp_merge = df[['SK_ID_CURR']]
        tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
        df['count_ratio_payment_' + str(i)] = tmp_merge['des'].fillna(0)

tmp = install.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'])['DAYS_INSTALMENT'].count().reset_index()

tmp = tmp[tmp['DAYS_INSTALMENT'] > 1]

tmp.columns = ['SK_ID_PREV','NUM_INSTALMENT_NUMBER','count_dup']

install = install.merge(tmp, on = ['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], how='left')

dup_install = install[install['count_dup'] > 1]
dup_install.reset_index(drop=True, inplace = True)

tmp = install[(install['AMT_PAYMENT'] < install['AMT_INSTALMENT']) & (install['DAYS_ENTRY_PAYMENT'] < install['DAYS_INSTALMENT'])]
tmp['ratio'] = tmp['AMT_PAYMENT']/tmp['AMT_INSTALMENT']
tmp = dup_install.groupby(['SK_ID_CURR'])['AMT_PAYMENT'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp_1.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp_1, on=['SK_ID_CURR'], how='left')
    df['min_special_install'] = tmp_merge['des1']
    df['max_special_install'] = tmp_merge['des2']
    df['mean_special_install'] = tmp_merge['des3']
    df['sum_special_install'] = tmp_merge['des4']
dup_install.sort_values(by=['SK_ID_PREV','NUM_INSTALMENT_NUMBER','DAYS_ENTRY_PAYMENT'])
credit = pd.read_csv("../input/credit_card_balance.csv")
credit['tmp'] = credit['AMT_BALANCE']/credit['AMT_CREDIT_LIMIT_ACTUAL']
tmp = credit.groupby(["SK_ID_CURR","SK_ID_PREV"])['tmp'].max().reset_index()
tmp = tmp.groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_max_ratio_balance_limit_credit'] = tmp_merge['des1']
    df['max_max_ratio_balance_limit_credit'] = tmp_merge['des2']
    df['mean_max_ratio_balance_limit_credit'] = tmp_merge['des3']
    df['sum_max_ratio_balance_limit_credit'] = tmp_merge['des4']

credit['tmp'] = credit['AMT_BALANCE']/credit['AMT_CREDIT_LIMIT_ACTUAL']
tmp = credit.groupby(["SK_ID_CURR","SK_ID_PREV"])['tmp'].min().reset_index()
tmp = credit.groupby(['SK_ID_CURR'])['tmp'].agg({"returns": [np.min, np.max,np.mean, np.sum]})\
                                                                .reset_index()
tmp.columns = ['SK_ID_CURR','des1','des2','des3', 'des4']
for df in [train,test]:
    tmp_merge = df[['SK_ID_CURR']]
    tmp_merge = tmp_merge.merge(tmp, on=['SK_ID_CURR'], how='left')
    df['min_min_ratio_balance_limit_credit'] = tmp_merge['des1']
    df['max_min_ratio_balance_limit_credit'] = tmp_merge['des2']
    df['mean_min_ratio_balance_limit_credit'] = tmp_merge['des3']
    df['sum_min_ratio_balance_limit_credit'] = tmp_merge['des4']

doc = [x for x in train.columns if 'FLAG_DOC' in x]
connection = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
       'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',]

le = LabelEncoder()
categorical = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_EDUCATION_TYPE',
              'NAME_FAMILY_STATUS',
               'FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_CONT_MOBILE','FLAG_EMAIL','FLAG_WORK_PHONE',
              'OCCUPATION_TYPE','ORGANIZATION_TYPE_v2',
              'NAME_INCOME_TYPE','NAME_HOUSING_TYPE','NAME_TYPE_SUITE',
              'NAME_CONTRACT_TYPE']
for i in  categorical:
    train[i.lower()] = le.fit_transform(train[i].fillna("NA"))
    test[i.lower()] = le.transform(test[i].fillna("NA"))
for df in [train,test]:   
    df['ratio_credit_annity'] = df['AMT_CREDIT']/df['AMT_ANNUITY']
    df['ratio_credit_goods'] = df['AMT_CREDIT']/df['AMT_GOODS_PRICE']
    #df['ratio_app_first_second'] = df['1st_recent_app']/df['2nd_recent_app']
    
    df['ratio_min_annuity'] = df['AMT_INCOME_TOTAL']/df['min_amt_annuity']
    df['ratio_max_annuity'] = df['AMT_INCOME_TOTAL']/df['max_amt_annuity']
    df['ratio_mean_annuity'] = df['AMT_INCOME_TOTAL']/df['mean_amt_annuity']
    
    tmp = df[df['NAME_CONTRACT_TYPE'] == "Revolving loans"].index
    df['ratio_credit_goods'].iloc[tmp] = np.nan
    
    df['ratio_credit_annity'].replace(20, np.nan, inplace = True)
    
    #tmp = df[df['max_endate_bureau'] < 0].index
    #df['max_endate_bureau'].iloc[tmp] = np.nan
    
    
    
    df['doc'] = df[doc].mean(axis=1)
    
    df['count_null_cash_loans'].replace(np.nan, 0, inplace = True)
    df['count_null_revolving_loans'].replace(np.nan, 0, inplace = True)
    #df['OWN_CAR_AGE'].fillna(-1, inplace = True)
    
    df['ratio_cntpay_cur_mean'] = df['ratio_credit_annity']/df['mean_cntpay']
    df['ratio_cntpay_cur_min'] = df['ratio_credit_annity']/df['min_cntpay']
    df['ratio_cntpay_cur_max'] = df['ratio_credit_annity']/df['max_cntpay']
    
    df['delta_bureau_HC'] = df['max_day_lastdue'] - df['max_endate_bureau']
    df['frequency_bureau'] = (df['max_endate_bureau'] - df['min_endate_bureau'])/(df['count_active_bureau_v2'])
    df['frequency_bureau'].replace(0, np.nan)
    #df['master_sum'] = df['AMT_CREDIT'] + df['sum_notfinish']
    
    #df['frequency'] = (df['max_day_decision'] - df['min_day_decision'])/(df['count_cash_loans'] + df['count_consumer_loans']+ df['count_revolving_loans'])
    #df['ratio_annuity'] = df['mean_amt_app']/df['mean_amt_annuity']
    #df['ratio_annuity_v1'] df['mean_amt_app_v1']/df['mean_amt_annuity_v1']
    df['sum_delta_install_credit_curr'] = df['AMT_CREDIT'] + df['sum_delta_paid_install']
    
    df['strenght_income'] = df['sum_delta_install_credit_curr']/df['AMT_INCOME_TOTAL']
    
    df['sum_notfinish'] = df['sum_notfinish_cash_loans'] + df['sum_notfinish_consumer_loans']
    df['ratio_income_notfinish'] = df['sum_notfinish']/df['AMT_CREDIT']
    df['connection'] = df[connection].mean(axis=1)
    df['living'] = df[['REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
       'LIVE_CITY_NOT_WORK_CITY']].mean(axis=1)
    
predictors = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','CNT_CHILDREN',
              'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
              'CNT_FAM_MEMBERS', 'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
              'DAYS_LAST_PHONE_CHANGE',
              'OBS_30_CNT_SOCIAL_CIRCLE', 
              'DEF_30_CNT_SOCIAL_CIRCLE',
              'OWN_CAR_AGE','REGION_POPULATION_RELATIVE',
              #'AMT_REQ_CREDIT_BUREAU_MON',
              'AMT_REQ_CREDIT_BUREAU_YEAR',
              'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','TOTALAREA_MODE','APARTMENTS_AVG',
             #'FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE',
      #'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
              'ratio_credit_annity','ratio_credit_goods','doc','connection',
              'min_amt_app','max_amt_app','mean_amt_app',
              'min_amt_app_v1','max_amt_app_v1','mean_amt_app_v1',
              'min_amt_card','max_amt_card','mean_amt_card',
        
              #'min_amt_credit_master','max_amt_credit_master','mean_amt_credit_master',
              #'ratio_1st_min','ratio_1st_max','ratio_1st_mean',
              #'ratio_min_annuity','ratio_max_annuity','ratio_mean_annuity',
              
              'min_amt_app_fail','max_amt_app_fail','mean_amt_app_fail',
              'min_amt_app_v1_fail','max_amt_app_v1_fail','mean_amt_app_v1_fail',
              'min_amt_card_fail','max_amt_card_fail','mean_amt_card_fail',
              
             # 'min_amt_app_fail_v1','max_amt_app_fail_v1','mean_amt_app_fail_v1',
              #'min_amt_app_v1_fail_v1','max_amt_app_v1_fail_v1','mean_amt_app_v1_fail_v1',
             # 'min_amt_card_fail_v1','max_amt_card_fail_v1','mean_amt_card_fail_v1',
              
              'min_amt_annuity', 'max_amt_annuity', 'mean_amt_annuity',
              #'min_amt_annuity_v1', 'max_amt_annuity_v1', 'mean_amt_annuity_v1',
              
              'min_cntpay', 'max_cntpay', 'mean_cntpay',# 'CNT_PAYMENT',
              
              'min_day_decision','max_day_decision',
              'min_day_termination','max_day_termination',
              #'mean_day_termination',
              'max_day_lastdue',#'2nd_day_lastdue',
              'min_firstdraw', 'max_firstdraw', #'mean_firstdraw',
              'min_firstdraw_decision', 'max_firstdraw_decision', 'mean_firstdraw_decision',
              'min_firstdraw_firstdue', 'max_firstdraw_firstdue', 'mean_firstdraw_firstdue',
              'min_firstdraw_lastdue', 'max_firstdraw_lastdue', 'mean_firstdraw_lastdue',
              'max_day_decision_fail',
              'count_notfinish_revolving_loans','count_notfinish_cash_loans','count_notfinish_consumer_loans',
              'min_sooner', 'max_sooner', 'mean_sooner',
              'min_seller', 'max_seller', 'mean_seller',
              'sum_notfinish',#'master_sum',
              
     
              'min_amt_goods_v1','max_amt_goods_v1','mean_amt_goods_v1',
              
              '1st_recent_app','1st_recent_credit','1st_recent_card',#'1st_recent_app_v1','1st_recent_credit_v1'
             '1st_recent_app_fail','1st_recent_credit_fail','1st_recent_card_fail',
              '1st_recent_ratedown','1st_recent_ratedown_fail',
              '1st_recent_cntpay',
              
              'ratio_cntpay_cur_mean',
          
              'count_cash_loans','count_consumer_loans','count_clear_reason',
              'count_middle', 'count_low_normal', 'count_high', 'count_low_action',
              'ratio_middle', 'ratio_low_normal', 'ratio_high', 'ratio_low_action',
              
              ##BUREAU
              
              'count_active_bureau','count_closed_bureau',
              'count_active_bureau_v2',#'count_closed_bureau_v2',
              #'count_active_bureau_v3',#'count_closed_bureau_v3',
              'min_active_credit_bureau','max_active_credit_bureau',
              'mean_active_credit_bureau', #'sum_active_credit_bureau',
            
               'min_closed_credit_bureau','max_closed_credit_bureau',
              'mean_closed_credit_bureau', #'sum_closed_credit_bureau',
              
    
              'min_active_credit_bureau_v1','max_active_credit_bureau_v1',
              'mean_active_credit_bureau_v1', #'sum_active_credit_bureau',
            
              'min_active_credit_bureau_v2','max_active_credit_bureau_v2',
              'mean_active_credit_bureau_v2', 
             
              
              'min_active_credit_bureau_v3','max_active_credit_bureau_v3',
              'mean_active_credit_bureau_v3',
              #'min_startdate_bureau','max_startdate_bureau',
              'max_endate_bureau',
              '1st_endate_bureau',#'2nd_endate_bureau',
              #'max_endate_bureau_v1',
              #'1st_endate_bureau_v1',
              'max_endatefact_bureau',#'1st_endatefact_bureau',
              'min_deltaendate_bureau','max_deltaendate_bureau','mean_deltaendate_bureau',
              #'min_deltaendate_bureau_v1','max_deltaendate_bureau_v1','mean_deltaendate_bureau_v1',
              'min_duration_bureau','max_duration_bureau','mean_duration_bureau',
              #'min_duration_bureau_v1','max_duration_bureau_v1','mean_duration_bureau_v1',
              'min_sooner_bureau','max_sooner_bureau','mean_sooner_bureau',
              #'min_sooner_bureau_v1','max_sooner_bureau_v1','mean_sooner_bureau_v1',
              'min_annuity_bureau','max_annuity_bureau','mean_annuity_bureau',
              
              'min_debt_bureau','max_debt_bureau','mean_debt_bureau','sum_debt_bureau',
              'min_debt_bureau_v1','max_debt_bureau_v1','mean_debt_bureau_v1',
              'min_limit_bureau','max_limit_bureau','mean_limit_bureau',
             # 'min_limit_bureau_v1','max_limit_bureau_v1','mean_limit_bureau_v1',
              'min_overdue_bureau','max_overdue_bureau','mean_overdue_bureau',
              #'min_overdue_bureau_v1','max_overdue_bureau_v1','mean_overdue_bureau_v1',
              'min_ratio_debt_credit_bureau','max_ratio_debt_credit_bureau','mean_ratio_debt_credit_bureau',
              
              ##INSTALMENT
              'min_delta_num_install','max_delta_num_install','mean_delta_num_install',
              'min_ratio_num_install','max_ratio_num_install','mean_ratio_num_install',
              'min_max_version_install','max_max_version_install','mean_max_version_install',
              'min_ratio_paid_install','max_ratio_paid_install','mean_ratio_paid_install',
              'sum_delta_paid_install','sum_delta_install_credit_curr',#'strenght_income'
              'min_max_num_install','max_max_num_install','mean_max_num_install',

              'count_small_payment','count_late_payment_0','count_late_payment_10','count_late_payment_20','count_late_payment_30',
              'min_max_ratio_balance_limit_credit','max_max_ratio_balance_limit_credit','mean_max_ratio_balance_limit_credit',
              #'min_min_ratio_balance_limit_credit','max_min_ratio_balance_limit_credit','mean_min_ratio_balance_limit_credit'
              #'count_dbd_def_pos'
              #'min_special_install','max_special_install','mean_special_install','sum_special_install'
              #'min_recent_late_install',
            #'count_small_payment_v2','count_late_payment_v2','count_late_payment_lv1_v2','count_late_payment_lv2_v2','count_late_payment_lv3_v2',
                #'min_delta_days_install','max_delta_days_install','mean_delta_days_install'
              #'count_special'
              #'min_std_install','max_std_install','mean_std_install'
              # 'min_later_install','max_later_install','mean_later_install'
              #'min_first_late_install','max_first_late_install','mean_first_late_install'
              #'min_count_late_payment','max_count_late_payment','mean_count_late_payment'
              #'min_count_small_payment','max_count_small_payment','mean_count_small_payment'
              #'min_max_amt_install','max_max_amt_install','mean_max_amt_install'
             # 'min_first2_install','max_first2_install','mean_first2_install'
             # 'min_num_version_install','max_num_version_install','mean_num_version_install'
              #'mean_count_bb'
             #'min_max_install','max_max_install','mean_max_install',
              #'min_count_install','max_count_install','mean_count_install',
              #'min_ratio_max_cnt_install','max_ratio_max_cnt_install','mean_ratio_max_cnt_install',
              #'min_delta_max_cnt_install','max_delta_max_cnt_install','mean_delta_max_cnt_install',
              #"min_delta_count_max_install_v2","max_delta_count_max_install_v2","mean_delta_count_max_install_v2"
             ] +\
             [i.lower() for i in categorical]
categorical = [i.lower() for i in categorical]

NFOLDS = 5
kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2018)
pred_test_full = 0
params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 40,  
    'max_depth': 7,  
    'colsample_bytree': 0.15,  
    'seed': 101
        }

res = []
idx = 0
for dev_index, val_index in kf.split(train, train['TARGET'].values): 
    #idx += 1
    #if (idx != 3):
    #    continue
    dev, valid = train.loc[dev_index,:], train.loc[val_index,:]
    dtrain = lgb.Dataset(dev[predictors].values, label=dev['TARGET'].values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )
    dvalid = lgb.Dataset(valid[predictors].values, label=valid['TARGET'].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    print("Training the model...")
    lgb_model = lgb.train(params, 
                     dtrain, 
                     valid_sets=[dtrain, dvalid], 
                     valid_names=['train','valid'], 
                     num_boost_round= 30000,
                         early_stopping_rounds=500,
                         verbose_eval=100, 
                         feval=None)
    oof = pd.DataFrame()
    oof['id'] = valid['SK_ID_CURR'].values
    oof['target'] = valid['TARGET'].values
    oof['preds'] = lgb_model.predict(valid[predictors],num_iteration=lgb_model.best_iteration)
    res.append(oof)
    pred_test_full += lgb_model.predict(test[predictors],num_iteration=lgb_model.best_iteration)
    
    


sub = pd.read_csv("../input/sample_submission.csv")

sub['TARGET'] = pred_test_full/NFOLDS

sub.to_csv("sub_lgb.csv", index=False)

res = pd.concat(res, ignore_index=True)
res.to_csv("oof_lgb.csv", index=False)
print(roc_auc_score(res['target'], res['preds']))
  
tmp = install.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'])['DAYS_INSTALMENT'].count().reset_index()

tmp = tmp[tmp['DAYS_INSTALMENT'] > 1]

tmp.columns = ['SK_ID_PREV','NUM_INSTALMENT_NUMBER','count_dup']

install = install.merge(tmp, on = ['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], how='left')

tmp = install[install['count_dup'] > 1]
install.drop(['count_dup_x','count_dup_y'], axis=1, inplace = True)
tmp.sort_values(by=['SK_ID_PREV','NUM_INSTALMENT_NUMBER','DAYS_ENTRY_PAYMENT'])
tmp['SK_ID_CURR'].value_counts()
