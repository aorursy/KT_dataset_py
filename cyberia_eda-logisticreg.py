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
train_identity_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_transaction_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

test_identity_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_transaction_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

ssub_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
ssub_df.sort_values(by = ['TransactionID']).tail()
test_identity_df.shape
ssub_df.nunique()
train_transaction_df.shape
import matplotlib.pyplot as plt

from matplotlib_venn import venn2
venn2([set(train_identity_df.TransactionID), set(train_transaction_df)])
venn2([set(test_identity_df.TransactionID), set(test_transaction_df)])
train_transaction_df.groupby(['TransactionID']).count()
train_identity_df.describe(include = 'all')
train_ = pd.merge(train_identity_df, train_transaction_df, how = 'inner', on = 'TransactionID')
train_.shape
test_ = pd.merge(test_identity_df, test_transaction_df, how = 'inner', on = 'TransactionID')
venn2([set(train_.TransactionID), set(test_.TransactionID)])
train_.head()
test_.head()
train_.shape
test_.shape
train_fil = train_[train_.TransactionID.isin(test_.TransactionID)]
train_fil.shape
train_.tail()
test_.head()
# train_transaction_df['isFraud'] = train_transaction_df.isFraud.astype('category')
train_transaction_df.isFraud.dtype
from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs')

clf.fit(train_transaction_df.TransactionAmt, train_transaction_df.isFraud)
plt.scatter(train_transaction_df.TransactionAmt, train_transaction_df.isFraud)
smaller_df = train_transaction_df.head(500)
import seaborn as sns; sns.set(color_codes = True)
sns.regplot(x = smaller_df.TransactionAmt, y = smaller_df.isFraud, logistic=True)
sns.regplot(x = train_transaction_df.TransactionAmt, y = train_transaction_df.isFraud, logistic=True)