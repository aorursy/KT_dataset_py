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
df_train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

df_train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')



df_test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

df_test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')



df_sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
df_train = pd.merge(df_train_transaction, df_train_identity, on='TransactionID', how='left')

df_test = pd.merge(df_test_transaction, df_test_identity, on='TransactionID', how='left')
df_na_rate = pd.DataFrame(df_train.isna().mean().sort_values(ascending=False), columns=['na_rate']).reset_index()

na_rate_columns = list(df_na_rate.loc[df_na_rate.na_rate >  0.4]['index'])
df_train=df_train.drop(columns=na_rate_columns)

df_test=df_test.drop(columns=na_rate_columns)
X_train = df_train.drop(columns=['isFraud'])

y_train = df_train['isFraud']

X_test =df_test
X_test_original = X_test.copy()

X_train = X_train.loc[:, 'C1':]

X_test = X_test.loc[:, 'C1':]
X_train = X_train.drop(columns=['M6'])

X_test = X_test.drop(columns=['M6'])
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
X_train.fillna(-999, inplace=True)

X_test.fillna(-999, inplace=True)

model.fit(X_train, y_train)
p= model.predict(X_test)

p_prob = model.predict_proba(X_test)
p_prob = p_prob[:,1]
p_prob
result = pd.Series(p_prob, index=X_test_original['TransactionID'], name='isFraud')
result.to_csv('submission.csv', header=True)