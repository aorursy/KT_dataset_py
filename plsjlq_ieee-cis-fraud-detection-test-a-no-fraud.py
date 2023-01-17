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
train_transaction=pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

train_identity=pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

test_transaction=pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

test_identity=pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

print("Data is loaded")
print(train_transaction.shape)

print(train_identity.shape)

print(test_transaction.shape)

print(test_identity.shape)
numOf_fraud=train_transaction['isFraud'].sum()

print(numOf_fraud)
per_Fraud = numOf_fraud/len(train_transaction)*100

print(f'about {round(per_Fraud,2)}% of transactions are fraud')
pred_non_F=np.zeros((506691,1))

print(pred_non_F.shape)
sample_submission['isFraud'] = pred_non_F

sample_submission.to_csv('sample_submission.csv', index=True)

sample_submission