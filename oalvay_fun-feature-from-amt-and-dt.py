# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

import time

import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
def time2next_sameAmt(train):

    time2next = []

    time2prev = []

    start = time.time()

    train_range = range(len(train))

    for i in train_range:

        temp = list(train[train['TransactionAmt'] == train.iloc[i].TransactionAmt].index)

        try: 

            j = temp[temp.index(i) + 1]

            time2next.append(train.iloc[j].TransactionDT - train.iloc[i].TransactionDT)

        except: 

            if len(temp) == 1:

                time2next.append(-1)

            else:

                time2next.append(-2)

        try:

            k = temp[temp.index(i) - 1]

            time2prev.append(train.iloc[i].TransactionDT - train.iloc[k].TransactionDT)

        except:     

            # this part is meaningless as list[-1] is not a error. I will solve this problem later.

            if len(temp) == 1:      

                time2prev.append(-1) 

            else:

                time2prev.append(-2)

    end = time.time()

    print('time: ', round(end - start, 1))

    return time2next, time2prev
#test['time2next'], test['time2prev'] = time2next_sameAmt(test)

#train['time2next'], train['time2prev'] = time2next_sameAmt(train)
train_t2sameAmt = pd.read_csv('../input/ieee-time2next-sameamt/train_t2sameAmt.csv')

test_t2sameAmt = pd.read_csv('../input/ieee-time2next-sameamt/test_t2sameAmt.csv')



train = pd.merge(train, train_t2sameAmt, on='TransactionID', how='left')

train = train[['isFraud', 'time2prev', 'time2next', 'TransactionAmt']].copy()



# analysis usage, don't need test dataset here:

#test = pd.merge(test, test_t2sameAmt, on='TransactionID', how='left')

#test = test.drop(['time2prev', 'time2next'], axis = 1)
def local_freq_encode(col, train):

    return train[col].map(train[col].value_counts().to_dict())



train['TransactionAmt_fq_enc']= local_freq_encode('TransactionAmt', train)

train.loc[train.time2prev <0, 'time2prev'] = -2

train.loc[train.TransactionAmt_fq_enc == 1, 'time2prev'] = -1

# test['TransactionAmt_fq_enc'] = local_freq_encode('TransactionAmt', test)

# test.loc[test.time2prev <0, 'time2prev'] = -2

# test.loc[test.TransactionAmt_fq_enc == 1, 'time2prev'] = -1
sns.set(rc={'figure.figsize':(16.7,8.27)})

sns.distplot(train.loc[(train['isFraud'] == 0) & (train.time2next < 201) & (train.time2next > 0),

                       'time2next'], color = 'blue', kde = False, bins = 200)

plt.title('Histogram for feature time2next \nin non-Fraud transaction', size = 20 )

plt.show()
sns.set(rc={'figure.figsize':(16.7,8.27)})

sns.distplot(train.loc[(train['isFraud'] == 1) & (train.time2next < 201) & (train.time2next > 0),

                       'time2next'], color = 'red', kde = False, bins = 200)



plt.title('Histogram for feature time2next \nin Fraud transaction', size = 20 )

plt.show()