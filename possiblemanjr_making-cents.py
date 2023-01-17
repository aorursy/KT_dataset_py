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
########################### DATA LOAD

#################################################################################

print('Load Data')

train_df = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_df = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

test_df['isFraud'] = 0



train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
train_df['cents'] = np.round( train_df['TransactionAmt'] - np.floor(train_df['TransactionAmt']),2 )

test_df['cents'] = np.round( test_df['TransactionAmt'] - np.floor(test_df['TransactionAmt']),2 )

train_df[train_df.ProductCD=='W'].groupby('cents')['isFraud'].agg(['mean','count']).sort_values('count',ascending=False).iloc[:5]
cents_train = train_df[['TransactionID','cents']]

cents_test = test_df[['TransactionID','cents']]
cents_train
cents_train
cents_train.to_csv('cents_train.csv')

cents_test.to_csv('cents_test.csv')


