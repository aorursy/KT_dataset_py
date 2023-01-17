import pandas as pd

import numpy as np

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import xgboost as xgb

import itertools

from pprint import pprint

import random

import os

import gc

from time import time

import datetime

from tqdm import tqdm

from tqdm import tqdm_notebook



from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline



def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



SEED = 1993

seed_everything(SEED)

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
# reduce_mem_usage()

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
files = ['../input/ieee-fraud-detection/test_identity.csv', 

         '../input/ieee-fraud-detection/test_transaction.csv',

         '../input/ieee-fraud-detection/train_identity.csv',

         '../input/ieee-fraud-detection/train_transaction.csv',

         '../input/ieee-fraud-detection/sample_submission.csv']
%%time

def load_data(file):

    return pd.read_csv(file)



with multiprocessing.Pool() as pool:

    test_identity, test_transaction, train_identity, train_transaction, sample_submission = pool.map(load_data, files)
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test  = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

sample_submission = sample_submission.set_index('TransactionID')



print("Train shape: ", train.shape)

print("Test shape:", test.shape)



y = train['isFraud'].copy()

del train_transaction, train_identity, test_transaction, test_identity

gc.collect()



# Drop target, fill in NaNs

train = train.drop('isFraud', axis=1)
train = reduce_mem_usage(train)

test  = reduce_mem_usage(test)
useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',

                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',

                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',

                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',

                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',

                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',

                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',

                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',

                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',

                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',

                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',

                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',

                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',

                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',

                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',

                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',

                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',

                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',

                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',

                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
cols_to_drop = [col for col in train.columns if col not in useful_features]

cols_to_drop.remove('TransactionID')

cols_to_drop.remove('TransactionDT')
print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))



train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
def addNewFeatures(data): 

    data['uid1'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)

    data['uid2'] = data['uid1'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)

    data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

    

    data['D9'] = np.where(data['D9'].isna(),0,1)

    

    return data



train = addNewFeatures(train)

test  = addNewFeatures(test)



# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

train['Transaction_hour_of_day'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_hour_of_day'] = np.floor(test['TransactionDT'] / 3600) % 24

train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24



train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))

    

for feature in ['id_34', 'id_36']:

    if feature in useful_features:

        # Count encoded for both train and test

        train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        

for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:

    if feature in useful_features:

        # Count encoded separately for train and test

        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
# https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering

train['card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

test['card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))



train['card2_count_full'] = train['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))

test['card2_count_full'] = test['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))



train['card3_count_full'] = train['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))

test['card3_count_full'] = test['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))



train['card4_count_full'] = train['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))

test['card4_count_full'] = test['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))



train['card5_count_full'] = train['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))

test['card5_count_full'] = test['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))



train['card6_count_full'] = train['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))

test['card6_count_full'] = test['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))





train['addr1_count_full'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))

test['addr1_count_full'] = test['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))



train['addr2_count_full'] = train['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))

test['addr2_count_full'] = test['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')



train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')



train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')
train['TransactionAmt'] = np.log1p(train['TransactionAmt'])

test['TransactionAmt']  = np.log1p(test['TransactionAmt'])
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')

us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())



def setTime(df):

    # Temporary variables for aggregation

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)

    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)

    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)

    

    df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)

    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)

    df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)

        

    # Possible solo feature

    df['is_december'] = df['DT'].dt.month

    df['is_december'] = (df['is_december']==12).astype(np.int8)



    # Holidays

    df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

    

    return df

    

train = setTime(train)

test  = setTime(test)
i_cols = ['card1','card2','card3','card5','uid1','uid2','uid3']



for col in i_cols:

    for agg_type in ['mean','std']:

        new_col_name = col+'_TransactionAmt_'+agg_type

        temp_df = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])

        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)

        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(

                                                columns={agg_type: new_col_name})



        temp_df.index = list(temp_df[col])

        temp_df = temp_df[new_col_name].to_dict()   



        train[new_col_name] = train[col].map(temp_df)

        test[new_col_name]  = test[col].map(temp_df)



train = train.replace(np.inf,999)

test  = test.replace(np.inf,999)
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100778

train['P_isproton'] = (train['P_emaildomain']=='protonmail.com')

train['R_isproton'] = (train['R_emaildomain']=='protonmail.com')

test['P_isproton']  = (test['P_emaildomain']=='protonmail.com')

test['R_isproton']  = (test['R_emaildomain']=='protonmail.com')
train['nulls1'] = train.isna().sum(axis=1)

test['nulls1'] = test.isna().sum(axis=1)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',

          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',

          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 

          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',

          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',

          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',

          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',

          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',

          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',

          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',

          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']
for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin']  = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix']  = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix']  = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
p = 'P_emaildomain'

r = 'R_emaildomain'

uknown = 'email_not_provided'



def setDomain(df):

    df[p] = df[p].fillna(uknown)

    df[r] = df[r].fillna(uknown)

    

    # Check if P_emaildomain matches R_emaildomain

    df['email_check'] = np.where((df[p]==df[r]) & (df[p]!=uknown),1,0)



    df[p+'_prefix'] = df[p].apply(lambda x: str(x).split('.')[0])

    df[r+'_prefix'] = df[r].apply(lambda x: str(x).split('.')[0])

    

    return df

    

train=setDomain(train)

test=setDomain(test)
train["lastest_browser"] = np.zeros(train.shape[0])

test["lastest_browser"]  = np.zeros(test.shape[0])



def setBrowser(df):

    df.loc[df["id_31"] == "samsung browser 7.0",'lastest_browser']=1

    df.loc[df["id_31"] == "opera 53.0",'lastest_browser']=1

    df.loc[df["id_31"] == "mobile safari 10.0",'lastest_browser']=1

    df.loc[df["id_31"] == "google search application 49.0",'lastest_browser']=1

    df.loc[df["id_31"] == "firefox 60.0",'lastest_browser']=1

    df.loc[df["id_31"] == "edge 17.0",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 69.0",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 67.0 for android",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 63.0 for android",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 63.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 64.0",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 64.0 for android",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 64.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 65.0",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 65.0 for android",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 65.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 66.0",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 66.0 for android",'lastest_browser']=1

    df.loc[df["id_31"] == "chrome 66.0 for ios",'lastest_browser']=1

    return df



train = setBrowser(train)

test  = setBrowser(test)
def setDevice(df):

    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

    

    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]



    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    

    df['had_id'] = 1

    gc.collect()

    

    return df



train = setDevice(train)

test  = setDevice(test)
i_cols = ['card1','card2','card3','card5',

          'C1','C2','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',

          'D1','D2','D3','D4','D5','D6','D8',

          'addr1','addr2',

          'dist1',

          'P_emaildomain', 'R_emaildomain',

          'DeviceInfo','device_name',

          'id_30','id_33',

          'uid1','uid2','uid3',

         ]



for col in i_cols:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   

    train[col+'_fq_enc'] = train[col].map(fq_encode)

    test[col+'_fq_enc']  = test[col].map(fq_encode)





for col in ['DT_M','DT_W','DT_D']:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts().to_dict()

            

    train[col+'_total'] = train[col].map(fq_encode)

    test[col+'_total']  = test[col].map(fq_encode)

        



periods = ['DT_M','DT_W','DT_D']

i_cols = ['uid1']

for period in periods:

    for col in i_cols:

        new_column = col + '_' + period

            

        temp_df = pd.concat([train[[col,period]], test[[col,period]]])

        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)

        fq_encode = temp_df[new_column].value_counts().to_dict()

            

        train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)

        test[new_column]  = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)

        

        train[new_column] /= train[period+'_total']

        test[new_column]  /= test[period+'_total']
def get_too_many_null_attr(data):

    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]

    return many_null_cols



def get_too_many_repeated_val(data):

    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    return big_top_value_cols



def get_useless_columns(data):

    too_many_null = get_too_many_null_attr(data)

    print("More than 90% null: " + str(len(too_many_null)))

    too_many_repeated = get_too_many_repeated_val(data)

    print("More than 90% repeated value: " + str(len(too_many_repeated)))

    cols_to_drop = list(set(too_many_null + too_many_repeated))

    #cols_to_drop.remove('isFraud')

    return cols_to_drop
cols_to_drop = get_useless_columns(train)
train = train.drop(cols_to_drop, axis=1)

test  = test.drop(cols_to_drop, axis=1)



print(train.shape)

print(test.shape)

print(y.shape)
numerical_cols = train.select_dtypes(exclude = 'object').columns

categorical_cols = train.select_dtypes(include = 'object').columns
# Label Encoding

for f in train.columns:

    if train[f].dtype.name =='object' or test[f].dtype.name =='object': 

        lbl = LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))
train = train.fillna(-999)

test = test.fillna(-999)
print(train.isnull().sum().max())

print(test.isnull().sum().max())
X = train.drop(['TransactionID', 'TransactionDT', 'DT'], axis=1)

X_test  = test.drop(['TransactionID', 'TransactionDT', 'DT'], axis=1)
# y = train['isFraud'].copy()



print("X_train :", X.shape)

print("y_train :", y.shape)

print("X_test :", X_test.shape)
class XGBGridSearch:

    """

    Source:

    https://www.kaggle.com/xhlulu/ieee-fraud-efficient-grid-search-with-xgboost

    """

    def __init__(self, param_grid, cv=3, verbose=0, 

                 shuffle=False, random_state=SEED):

        self.param_grid = param_grid

        self.cv = cv

        self.random_state = random_state

        self.verbose = verbose

        self.shuffle = shuffle

        

        self.average_scores = []

        self.scores = []

    

    def fit(self, X, y):

        self._expand_params()

        self._split_data(X, y)

            

        for params in tqdm(self.param_list, disable=not self.verbose):

            avg_score, score = self._run_cv(X, y, params)

            self.average_scores.append(avg_score)

            self.scores.append(score)

        

        self._compute_best()



    def _run_cv(self, X, y, params):

        """

        Perform KFold CV on a single set of parameters

        """

        scores = []

        

        for train_idx, val_idx in self.splits:

            clf = xgb.XGBClassifier(**params)



            X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]

            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            

            clf.fit(X_train, y_train)

            

            y_val_pred = clf.predict_proba(X_val)[:, 1]

            

            score = roc_auc_score(y_val, y_val_pred)

            scores.append(score)

            

            gc.collect()

        

        avg_score = sum(scores) / len(scores)

        return avg_score, scores

            

    def _split_data(self, X, y):

        kf = KFold(n_splits=self.cv, 

                   shuffle=self.shuffle, 

                   random_state=self.random_state)

        self.splits = list(kf.split(X, y))

            

    def _compute_best(self):

        """

        Compute best params and its corresponding score

        """

        idx_best = np.argmax(self.average_scores)

        self.best_score_ = self.average_scores[idx_best]

        self.best_params_ = self.param_list[idx_best]



    def _expand_params(self):

        """

        This method expands a dictionary of lists into

        a list of dictionaries (each dictionary is a single

        valid params that can be input to XGBoost)

        """

        keys, values = zip(*self.param_grid.items())

        self.param_list = [

            dict(zip(keys, v)) 

            for v in itertools.product(*values)

        ]
param_grid = {

    'n_estimators': [500, 1000],

    'missing': [-999],

    'random_state': [1993],

    'n_jobs': [1],

    'tree_method': ['gpu_hist'],

    'max_depth': [9],

    'learning_rate': [0.048, 0.05],

    'subsample': [0.85, 0.9],

    'colsample_bytree': [0.85, 0.9],

    'reg_alpha': [0, 0.1],

    'reg_lambda': [1, 0.9]

}



grid = XGBGridSearch(param_grid, cv=6, verbose=1)

grid.fit(X, y)



print("Best Score:", grid.best_score_)

print("Best Params:", grid.best_params_)
clf = xgb.XGBClassifier(**grid.best_params_)

clf.fit(X, y)
sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

sample_submission.to_csv('simple_xgboost.csv')
from IPython.display import FileLinks

FileLinks('.') # input argument is specified folder