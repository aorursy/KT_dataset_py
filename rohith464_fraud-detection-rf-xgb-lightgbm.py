# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import os

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import seaborn as sns

import multiprocessing

import gc

from sklearn import preprocessing

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix,roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV



import lightgbm as lgb

import xgboost as xgb

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
import pandas as pd

sample_submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")
print("Number of data points in train data of identity data :", train_identity.shape)

print('-'*50)



print("Number of data points in train data of transaction data:", train_transaction.shape)

print('-'*50)



print("Number of data points in test data of transaction data:", test_transaction.shape)

print('-'*50)



print("Number of data points in test data of identity data:", test_identity.shape)

print('-'*50)

print("The attributes of identity data :", train_identity.columns.values)

print('-'*100)

print("The attributes of transaction data :", train_transaction.columns.values)
#Combining both transaction and identity data

train_data = pd.merge(train_transaction,train_identity, on='TransactionID', how='left',left_index=True,right_index=True)

test_data = pd.merge(test_transaction,test_identity, on='TransactionID', how='left',left_index=True,right_index=True)
print("Number of data points in total train data :", train_data.shape)

print("Number of data points in total test data :", test_data.shape)
train_data.head()
test_data.head()
for i in range(1,10):

    i = str(i)

    test_data.rename(columns={'id-0'+i: 'id_0'+i}, inplace=True)
for i in range(10,39):

    i = str(i)

    test_data.rename(columns={'id-'+i: 'id_'+i}, inplace=True)
#fill in inf values with some constant

train_data = train_data.replace([np.inf,-np.inf],999)

test_data = test_data.replace([np.inf,-np.inf],999)
## REducing memory

train_data = reduce_mem_usage(train_data)

test_data = reduce_mem_usage(test_data)
del train_identity,test_identity,train_transaction,test_transaction
def addNewFeatures(data): 

    data['uid'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)



    data['uid2'] = data['uid'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)



    data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)



    data['D9'] = np.where(data['D9'].isna(),0,1)

    

    return data



train_data = addNewFeatures(train_data)

test_data = addNewFeatures(test_data)
i_cols = ['card1','card2','card3','card5','uid','uid2','uid3']



for col in i_cols:

    for agg_type in ['mean','std']:

        new_col_name = col+'_TransactionAmt_'+agg_type

        temp_df = pd.concat([train_data[[col, 'TransactionAmt']], test_data[[col,'TransactionAmt']]])

        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)

        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(

                                                columns={agg_type: new_col_name})



        temp_df.index = list(temp_df[col])

        temp_df = temp_df[new_col_name].to_dict()   



        train_data[new_col_name] = train_data[col].map(temp_df)

        test_data[new_col_name]  = test_data[col].map(temp_df)

train_data['card1_count_full'] = train_data['card1'].map(pd.concat([train_data['card1'], test_data['card1']], ignore_index=True).value_counts(dropna=False))

test_data['card1_count_full'] = test_data['card1'].map(pd.concat([train_data['card1'], test_data['card1']], ignore_index=True).value_counts(dropna=False))



train_data['card2_count_full'] = train_data['card2'].map(pd.concat([train_data['card2'], test_data['card2']], ignore_index=True).value_counts(dropna=False))

test_data['card2_count_full'] = test_data['card2'].map(pd.concat([train_data['card2'], test_data['card2']], ignore_index=True).value_counts(dropna=False))



train_data['card3_count_full'] = train_data['card3'].map(pd.concat([train_data['card3'], test_data['card3']], ignore_index=True).value_counts(dropna=False))

test_data['card3_count_full'] = test_data['card3'].map(pd.concat([train_data['card3'], test_data['card3']], ignore_index=True).value_counts(dropna=False))



train_data['card4_count_full'] = train_data['card4'].map(pd.concat([train_data['card4'], test_data['card4']], ignore_index=True).value_counts(dropna=False))

test_data['card4_count_full'] = test_data['card4'].map(pd.concat([train_data['card4'], test_data['card4']], ignore_index=True).value_counts(dropna=False))



train_data['card5_count_full'] = train_data['card5'].map(pd.concat([train_data['card5'], test_data['card5']], ignore_index=True).value_counts(dropna=False))

test_data['card5_count_full'] = test_data['card5'].map(pd.concat([train_data['card5'], test_data['card5']], ignore_index=True).value_counts(dropna=False))



train_data['card6_count_full'] = train_data['card6'].map(pd.concat([train_data['card6'], test_data['card6']], ignore_index=True).value_counts(dropna=False))

test_data['card6_count_full'] = test_data['card6'].map(pd.concat([train_data['card6'], test_data['card6']], ignore_index=True).value_counts(dropna=False))

train_data['addr1_count_full'] = train_data['addr1'].map(pd.concat([train_data['addr1'], test_data['addr1']], ignore_index=True).value_counts(dropna=False))

test_data['addr1_count_full'] = test_data['addr1'].map(pd.concat([train_data['addr1'], test_data['addr1']], ignore_index=True).value_counts(dropna=False))



train_data['addr2_count_full'] = train_data['addr2'].map(pd.concat([train_data['addr2'], test_data['addr2']], ignore_index=True).value_counts(dropna=False))

test_data['addr2_count_full'] = test_data['addr2'].map(pd.concat([train_data['addr2'], test_data['addr2']], ignore_index=True).value_counts(dropna=False))
import datetime

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')



def setTime(df):

    df['TransactionDT'] = df['TransactionDT'].fillna(df['TransactionDT'].median())

    # Temporary

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month

    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear

    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear

    

    df['DT_hour'] = df['DT'].dt.hour

    df['DT_day_week'] = df['DT'].dt.dayofweek

    df['DT_day'] = df['DT'].dt.day

    

    return df

    

train_data=setTime(train_data)

test_data=setTime(test_data)
train_data = train_data.drop('DT',axis=1)

test_data = test_data.drop('DT',axis=1)

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',

          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft',

          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',

          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft',

          'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other',

          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',

          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',

          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',

          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']



for c in ['P_emaildomain', 'R_emaildomain']:

    train_data[c + '_bin'] = train_data[c].map(emails)

    test_data[c + '_bin'] = test_data[c].map(emails)



    

    train_data[c + '_suffix'] = train_data[c].map(lambda x: str(x).split('.')[-1])

    test_data[c + '_suffix'] = test_data[c].map(lambda x: str(x).split('.')[-1])



    

    train_data[c + '_suffix'] = train_data[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test_data[c + '_suffix'] = test_data[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')



p = 'P_emaildomain'

r = 'R_emaildomain'

uknown = 'email_not_provided'



def setDomain(df):

    df[p] = df[p].fillna(uknown)

    df[r] = df[r].fillna(uknown)

    

    # Check if P_emaildomain matches R_emaildomain

    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)



    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])

    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])

    

    return df

    

train_data=setDomain(train_data)

test_data=setDomain(test_data)

#x_cv= setDomain(x_cv)
train_data['P_isproton']=(train_data['P_emaildomain']=='protonmail.com')

train_data['R_isproton']=(train_data['R_emaildomain']=='protonmail.com')

test_data['P_isproton']=(test_data['P_emaildomain']=='protonmail.com')

test_data['R_isproton']=(test_data['R_emaildomain']=='protonmail.com')
train_data["lastest_browser"] = np.zeros(train_data.shape[0])

test_data["lastest_browser"] = np.zeros(test_data.shape[0])

#x_cv["lastest_browser"] = np.zeros(x_cv.shape[0])



def setBrowser(df):

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



train_data=setBrowser(train_data)

test_data=setBrowser(test_data)
def id_split(dataframe):

  dataframe['DeviceInfo'] = dataframe['DeviceInfo'].fillna('unknown_device').str.lower()

  dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]



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
train_data = id_split(train_data)

test_data = id_split(test_data)

#x_cv = id_split(x_cv)
# New feature - log of transaction amount.



train_data['TransactionAmt_Log'] = np.log(train_data['TransactionAmt'])

test_data['TransactionAmt_Log'] = np.log(test_data['TransactionAmt'])

#x_cv['TransactionAmt_Log'] = np.log(x_cv['TransactionAmt'])



# New feature - decimal part of the transaction amount.



train_data['TransactionAmt_decimal'] = ((train_data['TransactionAmt'] - train_data['TransactionAmt'].astype(int)) * 1000).astype(int)

test_data['TransactionAmt_decimal'] = ((test_data['TransactionAmt'] - test_data['TransactionAmt'].astype(int)) * 1000).astype(int)

#x_cv['TransactionAmt_decimal'] = ((x_cv['TransactionAmt'] - x_cv['TransactionAmt'].astype(int)) * 1000).astype(int)

train_data['nulls'] = train_data.isna().sum(axis=1)

test_data['nulls'] = test_data.isna().sum(axis=1)

#x_cv['nulls'] = x_cv.isna().sum(axis=1)
train_data['id_02_to_mean_card1'] = train_data['id_02'] / train_data.groupby(['card1'])['id_02'].transform('mean')

train_data['id_02_to_mean_card4'] = train_data['id_02'] / train_data.groupby(['card4'])['id_02'].transform('mean')

train_data['id_02_to_std_card1'] = train_data['id_02'] / train_data.groupby(['card1'])['id_02'].transform('std')

train_data['id_02_to_std_card4'] = train_data['id_02'] / train_data.groupby(['card4'])['id_02'].transform('std')



test_data['id_02_to_mean_card1'] = test_data['id_02'] / test_data.groupby(['card1'])['id_02'].transform('mean')

test_data['id_02_to_mean_card4'] = test_data['id_02'] / test_data.groupby(['card4'])['id_02'].transform('mean')

test_data['id_02_to_std_card1'] = test_data['id_02'] / test_data.groupby(['card1'])['id_02'].transform('std')

test_data['id_02_to_std_card4'] = test_data['id_02'] / test_data.groupby(['card4'])['id_02'].transform('std')



train_data['D15_to_mean_card1'] = train_data['D15'] / train_data.groupby(['card1'])['D15'].transform('mean')

train_data['D15_to_mean_card4'] = train_data['D15'] / train_data.groupby(['card4'])['D15'].transform('mean')

train_data['D15_to_std_card1'] = train_data['D15'] / train_data.groupby(['card1'])['D15'].transform('std')

train_data['D15_to_std_card4'] = train_data['D15'] / train_data.groupby(['card4'])['D15'].transform('std')



test_data['D15_to_mean_card1'] = test_data['D15'] / test_data.groupby(['card1'])['D15'].transform('mean')

test_data['D15_to_mean_card4'] = test_data['D15'] / test_data.groupby(['card4'])['D15'].transform('mean')

test_data['D15_to_std_card1'] = test_data['D15'] / test_data.groupby(['card1'])['D15'].transform('std')

test_data['D15_to_std_card4'] = test_data['D15'] / test_data.groupby(['card4'])['D15'].transform('std')



train_data['D15_to_mean_addr1'] = train_data['D15'] / train_data.groupby(['addr1'])['D15'].transform('mean')

train_data['D15_to_mean_card4'] = train_data['D15'] / train_data.groupby(['card4'])['D15'].transform('mean')

train_data['D15_to_std_addr1'] = train_data['D15'] / train_data.groupby(['addr1'])['D15'].transform('std')

train_data['D15_to_std_card4'] = train_data['D15'] / train_data.groupby(['card4'])['D15'].transform('std')



test_data['D15_to_mean_addr1'] = test_data['D15'] / test_data.groupby(['addr1'])['D15'].transform('mean')

test_data['D15_to_mean_card4'] = test_data['D15'] / test_data.groupby(['card4'])['D15'].transform('mean')

test_data['D15_to_std_addr1'] = test_data['D15'] / test_data.groupby(['addr1'])['D15'].transform('std')

test_data['D15_to_std_card4'] = test_data['D15'] / test_data.groupby(['card4'])['D15'].transform('std')

def get_too_many_null_attr(data):

    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]

    return many_null_cols



def get_too_many_repeated_val(data):

    big_top_value_cols = [col for col in data.columns if data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    return big_top_value_cols



def get_useless_columns(data):

    too_many_null = get_too_many_null_attr(data)

    print("More than 90% null: " + str(len(too_many_null)))

    too_many_repeated = get_too_many_repeated_val(data)

    print("More than 90% repeated value: " + str(len(too_many_repeated)))

    cols_to_drop = list(set(too_many_null + too_many_repeated))

    

    return cols_to_drop
cols_to_drop = get_useless_columns(train_data)
cols_to_drop.remove('isFraud')
train_data = train_data.drop(cols_to_drop, axis=1)

test_data = test_data.drop(cols_to_drop, axis=1)
# Assigning the target variable.

Y = train_data['isFraud'].copy()
# Drop target and transactionID and transactionDT



train_data = train_data.drop('isFraud', axis=1)

#x_cv = x_cv.drop('isFraud', axis=1)



train_data = train_data.drop('TransactionID', axis=1)

test_data = test_data.drop('TransactionID', axis=1)

#x_cv = x_cv.drop('TransactionID', axis=1)



train_data = train_data.drop('TransactionDT', axis=1)

test_data = test_data.drop('TransactionDT', axis=1)

#x_cv = x_cv.drop('TransactionDT', axis=1)
print(train_data.shape)

print(test_data.shape)

print(Y.shape)
for c in train_data.columns:

    if train_data[c].dtype=='float16' or  train_data[c].dtype=='float32' or  train_data[c].dtype=='float64':

        train_data[c].fillna(train_data[c].mean())

        test_data[c].fillna(test_data[c].mean())
# Label Encoding

for f in train_data.columns:

    if train_data[f].dtype=='object' or test_data[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_data[f].values))

        train_data[f] = lbl.transform(list(train_data[f].values))

        test_data[f] = test_data[f].map(lambda s: '<unknown>' if s not in lbl.classes_ else s)

        lbl.classes_ = np.append(lbl.classes_, '<unknown>')

        test_data[f] = lbl.transform(list(test_data[f].values))



print('Labelling done.')
# Hyperparameter grid

param_grid = {

    'boosting_type': ['gbdt'],

    'num_leaves': [300,400,500,600,700],

    'learning_rate': list(np.linspace(0.005, 0.05,num=10)),

    'reg_alpha': list(np.linspace(0.3, 0.7,num=10)),

    'reg_lambda': list(np.linspace(0.5, 0.9,num=10)),

    'colsample_bytree': list(np.linspace(0.6, 1, num=10)),

    'metric':'auc',

    'scale_pos_weight': [30],

    'max_bin': [255]

}
#modelling

clf = lgb.LGBMClassifier(objective='binary',random_state=42,n_jobs=-1,max_depth=-1)

grid = RandomizedSearchCV(clf,param_grid,verbose=1,cv=3,n_iter=10)

grid.fit(train_data,Y)
grid.best_score_
y_preds = grid.predict_proba(test_data)[:,1] 
sample_submission['isFraud']=y_preds
sample_submission.to_csv('Lightgbm.csv', index=False)
sample_submission.head()