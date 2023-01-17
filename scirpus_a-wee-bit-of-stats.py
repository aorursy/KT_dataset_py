import datetime

import gc

import numpy as np

import pandas as pd
def make_day_feature(df, offset=0, tname='TransactionDT'):

    """

    Creates a day of the week feature, encoded as 0-6. 

    

    Parameters:

    -----------

    df : pd.DataFrame

        df to manipulate.

    offset : float (default=0)

        offset (in days) to shift the start/end of a day.

    tname : str

        Name of the time column in df.

    """

    # found a good offset is 0.58

    days = df[tname] / (3600*24)        

    encoded_days = np.floor(days-1+offset) % 7

    return encoded_days



def make_hour_feature(df, tname='TransactionDT'):

    """

    Creates an hour of the day feature, encoded as 0-23. 

    

    Parameters:

    -----------

    df : pd.DataFrame

        df to manipulate.

    tname : str

        Name of the time column in df.

    """

    hours = df[tname] / (3600)        

    encoded_hours = np.floor(hours) % 24

    return encoded_hours



START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

train['TransactionDateTime'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

train['TransactionDate'] = [x.date() for x in train['TransactionDateTime']]

train['TransactionHour'] = train.TransactionDT // 3600

train['TransactionHourOfDay'] = train['TransactionHour'] % 24

train['TransactionDay'] = train.TransactionDT // (3600 * 24)
y = train.isFraud

del train['isFraud']
trainidentity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train = train.merge(trainidentity,on='TransactionID')
train['isFraud'] = y
train.head()
avgtrain = train.copy()
for c in avgtrain.columns[:-1]:

    #print(c)

    x = avgtrain[[c,'isFraud']].groupby(c).isFraud.mean().reset_index(drop=False).rename(columns={'isFraud':c+'_isFraud'})

    avgtrain = avgtrain.merge(x,on=c,how='left')

    del avgtrain[c]

    gc.collect()
mn = train.isFraud.mean()

avgtrain = avgtrain.fillna(mn)
test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

test['TransactionDateTime'] = test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

test['TransactionDate'] = [x.date() for x in test['TransactionDateTime']]

test['TransactionHour'] = test.TransactionDT // 3600

test['TransactionHourOfDay'] = test['TransactionHour'] % 24

test['TransactionDay'] = test.TransactionDT // (3600 * 24)

testidentity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test = test.merge(testidentity,on='TransactionID')
train.head()
test.head()
cnttrain = train.copy()

for c in cnttrain.columns[:-1]:

    #print(c)

    x = test[c].value_counts().reset_index(drop=False).rename(columns={'index':c,c:c+'_cnt'})

    cnttrain = cnttrain.merge(x,on=c,how='left')

    del cnttrain[c]

    gc.collect()

cnttrain = cnttrain.fillna(0)
avgtrain.head()
cnttrain.head()
maxnumber = cnttrain[cnttrain.columns[1:]].max()

cnttrain[cnttrain.columns[1:]] = np.tanh(cnttrain[cnttrain.columns[1:]]/maxnumber)
a = pd.DataFrame()



for c in train.columns[:-1]:

    #print(c)

    a[c] = mn*(1-cnttrain[c+'_cnt'])+(cnttrain[c+'_cnt'])*avgtrain[c+'_isFraud']



a['isFraud'] = avgtrain.isFraud.values
a=a.fillna(train.isFraud.mean())
a.head()
from sklearn.metrics import roc_auc_score

cols = []

aucs = []

for c in a.columns[:-1]:

    cols.append(c)

    aucs.append((roc_auc_score(a.isFraud,a[[c]].sum(axis=1))))
x = pd.DataFrame()

x['cols'] = cols

x['auc'] = aucs

x = x.sort_values(by='auc', ascending=False).reset_index(drop=True)
x.head(60)