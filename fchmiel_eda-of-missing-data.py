# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)





# listed on the data page

categorical_features = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",

                        "addr1", "addr2", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", "M5",

                        "M6", "M7", "M8", "M9", "DeviceType", "DeviceInfo", "id_12", "id_13", "id_14",

                        "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22", "id_23",

                        "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32", 

                        "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"]
# looking at the head of the training data you can see a significant number of NaN values.



train.head()
isna = train.isna().sum(axis=1)

isna_test = test.isna().sum(axis=1)
plt.hist(isna, normed=True, bins=30, alpha=0.4, label='train')

plt.hist(isna_test, normed=True, bins=30, alpha=0.4, label='test')

plt.xlabel('Number of features which are NaNs')

plt.legend()
plt.plot(train['TransactionDT'], isna, 'r.', label='train', markersize=2)

plt.plot(test['TransactionDT'], isna_test, 'b.', label='test', markersize=2)

plt.legend()

plt.xlabel('Transaction DT')

plt.ylabel('Number of NaNs')



plt.axvline(1e7, color='gray', ls='--')

plt.axvline(2.2e7, color='gray', ls='--')

plt.axvline(2.5e7, color='gray', ls='--')

_ = plt.hist(isna[(train['TransactionDT']>1e7)], bins=50, normed=True, alpha=0.4, label='Train')

_ = plt.hist(isna_test[(test['TransactionDT']<2.2e7)], bins=50, normed=True, alpha=0.4, label='Test - early time')

_ = plt.hist(isna_test[(test['TransactionDT']>2.5e7)], bins=50, normed=True, alpha=0.4, label='Test - late time')



plt.legend()

plt.xlabel('Number of NaNs for training instance')
training_missing = train.isna().sum(axis=0) / train.shape[0] 

test_missing = test.isna().sum(axis=0) / test.shape[0] 

change = (training_missing / test_missing).sort_values(ascending=False)

change = change[change<1e6] # remove the divide by zero errors
change
fig, axs = plt.subplots(ncols=2)



train_vals = train["D15"].fillna(-999)

test_vals = test[test["TransactionDT"]>2.5e7]["D15"].fillna(-999) # values following the shift





axs[0].hist(train_vals, alpha=0.5, normed=True, bins=25)

    

axs[1].hist(test_vals, alpha=0.5, normed=True, bins=25)





fig.set_size_inches(7,3)

plt.tight_layout()
isna_df = pd.DataFrame({'missing_count':isna,'isFraud':train['isFraud']})
plt.plot(isna_df.groupby('missing_count').mean(), 'k.')

plt.ylabel('Fraction of fradulent transactions')

plt.xlabel('Number of missing variables')

plt.axhline(0)