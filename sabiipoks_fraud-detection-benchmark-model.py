# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.feature_selection import RFECV

import lightgbm as lgb

from tqdm import tqdm_notebook

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

import multiprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the dataset from the csv file using pandas

train_transaction_dataset = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

train_identity_dataset = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

test_transaction_dataset = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

test_identity_dataset = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

sample_submission_dataset = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
# View columns

print(train_transaction_dataset.columns)

print(train_identity_dataset.columns)

print(test_transaction_dataset.columns)

print(test_identity_dataset.columns)
print(train_transaction_dataset.shape)

print(train_identity_dataset.shape)

print(test_transaction_dataset.shape)

print(test_identity_dataset.shape)
# Merge transaction and identity dataset

train_dataset = train_transaction_dataset.merge(train_identity_dataset, how='left', left_index=True, right_index=True)

test_dataset = test_transaction_dataset.merge(test_identity_dataset, how='left', left_index=True, right_index=True)



print(train_dataset.shape)

print(test_dataset.shape)
del train_transaction_dataset, test_transaction_dataset, train_identity_dataset, test_identity_dataset
# source https://www.kaggle.com/krishonaveen/xtreme-boost-and-feature-engineering

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
train_dataset=reduce_mem_usage(train_dataset)

test_dataset=reduce_mem_usage(test_dataset)

sample_submission_dataset=reduce_mem_usage(sample_submission_dataset)

print('training set shape:', train_dataset.shape)

print('test set shape:', test_dataset.shape)
import gc



gc.collect()
# source https://www.kaggle.com/nroman/recursive-feature-elimination

# we will eliminate some of the features:



# 1. Features with only 1 unique value

one_value_cols = [col for col in train_dataset.columns if train_dataset[col].nunique() <= 1]

one_value_cols_test = [col for col in test_dataset.columns if test_dataset[col].nunique() <= 1]



# 2. Features with more than 90% missing values

many_null_cols = [col for col in train_dataset.columns if train_dataset[col].isnull().sum() / train_dataset.shape[0] > 0.9]

many_null_cols_test = [col for col in test_dataset.columns if test_dataset[col].isnull().sum() / test_dataset.shape[0] > 0.9]



# 3. Features with the top value appears more than 90% of the time

big_top_value_cols = [col for col in train_dataset.columns if train_dataset[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

big_top_value_cols_test = [col for col in test_dataset.columns if test_dataset[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]



cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))

cols_to_drop.remove('isFraud')

print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))



train_dataset = train_dataset.drop(cols_to_drop, axis=1)

test_dataset = test_dataset.drop(cols_to_drop, axis=1)
def id_split(dataframe):

    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]

    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]



    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]

    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]



    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]

    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]



    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]

    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]



    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    dataframe['had_id'] = 1

    gc.collect()

    

    return dataframe
train_dataset = id_split(train_dataset)

test_dataset = id_split(test_dataset)
train_dataset['P_email']=(train_dataset['P_emaildomain']=='xmail.com')

train_dataset['R_email']=(train_dataset['R_emaildomain']=='xmail.com')

test_dataset['P_email']=(test_dataset['P_emaildomain']=='xmail.com')

test_dataset['R_email']=(test_dataset['R_emaildomain']=='xmail.com')
train_dataset['null'] = train_dataset.isna().sum(axis=1)

test_dataset['null'] = test_dataset.isna().sum(axis=1)
a = np.zeros(train_dataset.shape[0])

train_dataset["lastest_browser"] = a

a = np.zeros(test_dataset.shape[0])

test_dataset["lastest_browser"] = a

def browser(df):

    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1

    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1

    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1

    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1

    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1

    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    return df

train_dataset=browser(train_dataset)

test_dataset=browser(test_dataset)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:

    train_dataset[c + '_bin'] = train_dataset[c].map(emails)

    test_dataset[c + '_bin'] = test_dataset[c].map(emails)

    

    train_dataset[c + '_suffix'] = train_dataset[c].map(lambda x: str(x).split('.')[-1])

    test_dataset[c + '_suffix'] = test_dataset[c].map(lambda x: str(x).split('.')[-1])

    

    train_dataset[c + '_suffix'] = train_dataset[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test_dataset[c + '_suffix'] = test_dataset[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
%%time

from sklearn.preprocessing import LabelEncoder



for col in test_dataset.columns:

    if test_dataset[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(train_dataset[col].astype(str).values) + list(test_dataset[col].astype(str).values))

        train_dataset[col] = le.transform(list(train_dataset[col].astype(str).values))

        test_dataset[col] = le.transform(list(test_dataset[col].astype(str).values))
%%time

train_dataset = reduce_mem_usage(train_dataset)

test_dataset = reduce_mem_usage(test_dataset)
# New feature - log of transaction amount. ()

train_dataset['TransactionAmt_Log'] = np.log(train_dataset['TransactionAmt'])

test_dataset['TransactionAmt_Log'] = np.log(test_dataset['TransactionAmt'])



# New feature - decimal part of the transaction amount.

train_dataset['TransactionAmt_decimal'] = ((train_dataset['TransactionAmt'] - train_dataset['TransactionAmt'].astype(int)) * 1000).astype(int)

test_dataset['TransactionAmt_decimal'] = ((test_dataset['TransactionAmt'] - test_dataset['TransactionAmt'].astype(int)) * 1000).astype(int)



# New feature - day of week in which a transaction happened.

train_dataset['Transaction_day_of_week'] = np.floor((train_dataset['TransactionDT'] / (3600 * 24) - 1) % 7)

test_dataset['Transaction_day_of_week'] = np.floor((test_dataset['TransactionDT'] / (3600 * 24) - 1) % 7)



# New feature - hour of the day in which a transaction happened.

train_dataset['Transaction_hour'] = np.floor(train_dataset['TransactionDT'] / 3600) % 24

test_dataset['Transaction_hour'] = np.floor(test_dataset['TransactionDT'] / 3600) % 24
important_cols = ['TransactionAmt', 'ProductCD',

'card1',

'card2',

'card3',

'card4',

'card5',

'card6',

'addr1',

'addr2',

'dist1',

'P_emaildomain',

'R_emaildomain',

'C1',

'C2',

'C4',

'C5',

'C6',

'C7',

'C8',

'C9',

'C10',

'C11',

'C12',

'C13',

'C14',

'D1',

'D2',

'D3',

'D4',

'D5',

'D6',

'D8',

'D9',

'D10',

'D11',

'D12',

'D13',

'D14',

'D15',

'M2',

'M3',

'M4',

'M5',

'M6',

'M8',

'M9',

'V4',

'V5',

'V12',

'V13',

'V19',

'V20',

'V30',

'V34',

'V35',

'V36',

'V37',

'V38',

'V44',

'V45',

'V47',

'V53',

'V54',

'V56',

'V57',

'V58',

'V61',

'V62',

'V70',

'V74',

'V75',

'V76',

'V78',

'V82',

'V83',

'V87',

'V91',

'V94',

'V96',

'V97',

'V99',

'V126',

'V127',

'V128',

'V130',

'V131',

'V139',

'V143',

'V149',

'V152',

'V160',

'V165',

'V170',

'V187',

'V189',

'V201',

'V203',

'V204',

'V207',

'V208',

'V209',

'V210',

'V212',

'V217',

'V221',

'V222',

'V234',

'V257',

'V258',

'V261',

'V264',

'V265',

'V266',

'V267',

'V268',

'V271',

'V274',

'V275',

'V277',

'V278',

'V279',

'V280',

'V282',

'V283',

'V285',

'V287',

'V289',

'V291',

'V292',

'V294',

'V306',

'V307',

'V308',

'V310',

'V312',

'V313',

'V314',

'V315',

'V317',

'V323',

'V324',

'V332',

'V333',

'id_01',

'id_02',

'id_05',

'id_06',

'id_09',

'id_13',

'id_14',

'id_17',

'id_19',

'id_20',

'id_30',

'id_31',

'id_33',

'id_38',

'DeviceType',

'DeviceInfo',

'device_name', 'device_version', 'OS_id_30', 'version_id_30', 'browser_id_31', 'version_id_31', 'screen_width', 'screen_height',

'P_email','R_email', 'lastest_browser', 'null', 'P_emaildomain_bin', 'P_emaildomain_suffix', 'R_emaildomain_bin', 'R_emaildomain_suffix',

'TransactionAmt_Log', 'TransactionAmt_decimal', 'Transaction_day_of_week', 'Transaction_hour'

]
print(len(important_cols))
train_dataset = train_dataset[important_cols + ['isFraud'] ]

test_dataset = test_dataset[important_cols]
train_dataset['card1_count_full'] = train_dataset['card1'].map(pd.concat([train_dataset['card1'], test_dataset['card1']], ignore_index=True).value_counts(dropna=False))

test_dataset['card1_count_full'] = test_dataset['card1'].map(pd.concat([train_dataset['card1'], test_dataset['card1']], ignore_index=True).value_counts(dropna=False))



train_dataset['card2_count_full'] = train_dataset['card2'].map(pd.concat([train_dataset['card2'], test_dataset['card2']], ignore_index=True).value_counts(dropna=False))

test_dataset['card2_count_full'] = test_dataset['card2'].map(pd.concat([train_dataset['card2'], test_dataset['card2']], ignore_index=True).value_counts(dropna=False))



train_dataset['card3_count_full'] = train_dataset['card3'].map(pd.concat([train_dataset['card3'], test_dataset['card3']], ignore_index=True).value_counts(dropna=False))

test_dataset['card3_count_full'] = test_dataset['card3'].map(pd.concat([train_dataset['card3'], test_dataset['card3']], ignore_index=True).value_counts(dropna=False))



train_dataset['card4_count_full'] = train_dataset['card4'].map(pd.concat([train_dataset['card4'], test_dataset['card4']], ignore_index=True).value_counts(dropna=False))

test_dataset['card4_count_full'] = test_dataset['card4'].map(pd.concat([train_dataset['card4'], test_dataset['card4']], ignore_index=True).value_counts(dropna=False))



train_dataset['card5_count_full'] = train_dataset['card5'].map(pd.concat([train_dataset['card5'], test_dataset['card5']], ignore_index=True).value_counts(dropna=False))

test_dataset['card5_count_full'] = test_dataset['card5'].map(pd.concat([train_dataset['card5'], test_dataset['card5']], ignore_index=True).value_counts(dropna=False))



train_dataset['card6_count_full'] = train_dataset['card6'].map(pd.concat([train_dataset['card6'], test_dataset['card6']], ignore_index=True).value_counts(dropna=False))

test_dataset['card6_count_full'] = test_dataset['card6'].map(pd.concat([train_dataset['card6'], test_dataset['card6']], ignore_index=True).value_counts(dropna=False))





train_dataset['addr1_count_full'] = train_dataset['addr1'].map(pd.concat([train_dataset['addr1'], test_dataset['addr1']], ignore_index=True).value_counts(dropna=False))

test_dataset['addr1_count_full'] = test_dataset['addr1'].map(pd.concat([train_dataset['addr1'], test_dataset['addr1']], ignore_index=True).value_counts(dropna=False))



train_dataset['addr2_count_full'] = train_dataset['addr2'].map(pd.concat([train_dataset['addr2'], test_dataset['addr2']], ignore_index=True).value_counts(dropna=False))

test_dataset['addr2_count_full'] = test_dataset['addr2'].map(pd.concat([train_dataset['addr2'], test_dataset['addr2']], ignore_index=True).value_counts(dropna=False))
train_dataset['TransactionAmt_to_mean_card1'] = train_dataset['TransactionAmt'] / train_dataset.groupby(['card1'])['TransactionAmt'].transform('mean')

train_dataset['TransactionAmt_to_mean_card4'] = train_dataset['TransactionAmt'] / train_dataset.groupby(['card4'])['TransactionAmt'].transform('mean')

train_dataset['TransactionAmt_to_std_card1'] = train_dataset['TransactionAmt'] / train_dataset.groupby(['card1'])['TransactionAmt'].transform('std')

train_dataset['TransactionAmt_to_std_card4'] = train_dataset['TransactionAmt'] / train_dataset.groupby(['card4'])['TransactionAmt'].transform('std')



test_dataset['TransactionAmt_to_mean_card1'] = test_dataset['TransactionAmt'] / test_dataset.groupby(['card1'])['TransactionAmt'].transform('mean')

test_dataset['TransactionAmt_to_mean_card4'] = test_dataset['TransactionAmt'] / test_dataset.groupby(['card4'])['TransactionAmt'].transform('mean')

test_dataset['TransactionAmt_to_std_card1'] = test_dataset['TransactionAmt'] / test_dataset.groupby(['card1'])['TransactionAmt'].transform('std')

test_dataset['TransactionAmt_to_std_card4'] = test_dataset['TransactionAmt'] / test_dataset.groupby(['card4'])['TransactionAmt'].transform('std')



train_dataset['id_02_to_mean_card1'] = train_dataset['id_02'] / train_dataset.groupby(['card1'])['id_02'].transform('mean')

train_dataset['id_02_to_mean_card4'] = train_dataset['id_02'] / train_dataset.groupby(['card4'])['id_02'].transform('mean')

train_dataset['id_02_to_std_card1'] = train_dataset['id_02'] / train_dataset.groupby(['card1'])['id_02'].transform('std')

train_dataset['id_02_to_std_card4'] = train_dataset['id_02'] / train_dataset.groupby(['card4'])['id_02'].transform('std')



test_dataset['id_02_to_mean_card1'] = test_dataset['id_02'] / test_dataset.groupby(['card1'])['id_02'].transform('mean')

test_dataset['id_02_to_mean_card4'] = test_dataset['id_02'] / test_dataset.groupby(['card4'])['id_02'].transform('mean')

test_dataset['id_02_to_std_card1'] = test_dataset['id_02'] / test_dataset.groupby(['card1'])['id_02'].transform('std')

test_dataset['id_02_to_std_card4'] = test_dataset['id_02'] / test_dataset.groupby(['card4'])['id_02'].transform('std')



train_dataset['D15_to_mean_card1'] = train_dataset['D15'] / train_dataset.groupby(['card1'])['D15'].transform('mean')

train_dataset['D15_to_mean_card4'] = train_dataset['D15'] / train_dataset.groupby(['card4'])['D15'].transform('mean')

train_dataset['D15_to_std_card1'] = train_dataset['D15'] / train_dataset.groupby(['card1'])['D15'].transform('std')

train_dataset['D15_to_std_card4'] = train_dataset['D15'] / train_dataset.groupby(['card4'])['D15'].transform('std')



test_dataset['D15_to_mean_card1'] = test_dataset['D15'] / test_dataset.groupby(['card1'])['D15'].transform('mean')

test_dataset['D15_to_mean_card4'] = test_dataset['D15'] / test_dataset.groupby(['card4'])['D15'].transform('mean')

test_dataset['D15_to_std_card1'] = test_dataset['D15'] / test_dataset.groupby(['card1'])['D15'].transform('std')

test_dataset['D15_to_std_card4'] = test_dataset['D15'] / test_dataset.groupby(['card4'])['D15'].transform('std')



train_dataset['D15_to_mean_addr1'] = train_dataset['D15'] / train_dataset.groupby(['addr1'])['D15'].transform('mean')

train_dataset['D15_to_mean_card4'] = train_dataset['D15'] / train_dataset.groupby(['card4'])['D15'].transform('mean')

train_dataset['D15_to_std_addr1'] = train_dataset['D15'] / train_dataset.groupby(['addr1'])['D15'].transform('std')

train_dataset['D15_to_std_card4'] = train_dataset['D15'] / train_dataset.groupby(['card4'])['D15'].transform('std')



test_dataset['D15_to_mean_addr1'] = test_dataset['D15'] / test_dataset.groupby(['addr1'])['D15'].transform('mean')

test_dataset['D15_to_mean_card4'] = test_dataset['D15'] / test_dataset.groupby(['card4'])['D15'].transform('mean')

test_dataset['D15_to_std_addr1'] = test_dataset['D15'] / test_dataset.groupby(['addr1'])['D15'].transform('std')

test_dataset['D15_to_std_card4'] = test_dataset['D15'] / test_dataset.groupby(['card4'])['D15'].transform('std')
X_train = train_dataset.drop(columns='isFraud', axis=1)

y_train = train_dataset['isFraud']



X_test = test_dataset



del train_dataset, test_dataset

gc.collect()
#fill in mean for floats

for c in X_train.columns:

    if X_train[c].dtype=='float16' or  X_train[c].dtype=='float32' or  X_train[c].dtype=='float64':

        X_train[c].fillna(X_train[c].mean())

        X_test[c].fillna(X_train[c].mean())



#fill in -111 for categoricals

X_train = X_train.fillna(-111)

X_test = X_test.fillna(-111)
X_train.head()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

xgb_sub=sample_submission_dataset.copy()

xgb_sub['isFraud'] = 0







xgb = XGBClassifier(alpha=4, base_score=0.5, booster='gbtree', colsample_bylevel=1,

                    colsample_bynode=1, colsample_bytree=0.9, gamma=0.1,

                    learning_rate=0.05, max_delta_step=0, max_depth=9,

                    min_child_weight=1, missing=-111, n_estimators=500, n_jobs=1,

                    nthread=None, objective='binary:logistic', random_state=0,

                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                    silent=None, subsample=0.9, tree_method='gpu_hist', verbosity=1,

                    eval_metric="auc")

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]

    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    xgb.fit(X_train_,y_train_)

    del X_train_,y_train_

    #pred=xgb.predict_proba(X_test)[:,1]

    val=xgb.predict_proba(X_valid)[:,1]

    del X_valid

    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))

    del val,y_valid

    #xgb_sub['isFraud'] = xgb_sub['isFraud']+pred/n_fold

    #del pred

    gc.collect()

    

del xgb



pred=xgb.predict_proba(X_test)[:,1]

xgb_sub['isFraud'] = pred
xgb_sub.to_csv('submission_cv.csv', index=False)
xgb = XGBClassifier(alpha=4, base_score=0.5, booster='gbtree', colsample_bylevel=1,

                    colsample_bynode=1, colsample_bytree=0.9, gamma=0.1,

                    learning_rate=0.05, max_delta_step=0, max_depth=9,

                    min_child_weight=1, missing=-111, n_estimators=500, n_jobs=1,

                    nthread=None, objective='binary:logistic', random_state=0,

                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                    silent=None, subsample=0.9, tree_method='gpu_hist', verbosity=1,

                    eval_metric="auc")



xgb.fit(X_train,y_train)



xgb_sub['isFraud'] = xgb.predict_proba(X_test)[:,1]
xgb_sub.to_csv('submission_full.csv', index=False)