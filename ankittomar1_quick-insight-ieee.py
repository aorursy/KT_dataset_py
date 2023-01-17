import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load the full dataset 

df_transaction_1 = pd.read_csv('../input/train_transaction.csv')

df_transaction_2 = pd.read_csv('../input/test_transaction.csv')



df_identity_1 = pd.read_csv('../input/train_identity.csv')

df_identity_2 = pd.read_csv('../input/test_identity.csv')
# Taking a sample of dataset 

df_trans_sample1 = df_transaction_1.sample(frac=0.1, replace=True, random_state=1)

df_trans_sample1.shape
# Taking a sample of dataset 

df_trans_sample2 = df_transaction_2.sample(frac=0.1, replace=True, random_state=1)

df_trans_sample2.shape
# merging the dataset

df_sample_transaction = pd.concat([df_trans_sample1, df_trans_sample2])

df_sample_transaction.head(2)
# Taking a sample of dataset 

df_id_sample1 = df_identity_1.sample(frac=0.1, replace=True, random_state=1)

df_id_sample1.shape
# Taking a sample of dataset 

df_id_sample2 = df_identity_2.sample(frac=0.1, replace=True, random_state=1)

df_id_sample2.shape
# merging the dataset

df_sample_identity = pd.concat([df_id_sample1, df_id_sample2])

df_sample_identity.head(2)
# looking on the dataset using pandas profiling 

df_sample_transaction.info()
# columnns in transaction 

df_sample_transaction['isFraud'].nunique() # 2

df_sample_transaction['isFraud'].unique() # 2

# NaN is also there 
# looking for identity dataset 

df_sample_identity.info()