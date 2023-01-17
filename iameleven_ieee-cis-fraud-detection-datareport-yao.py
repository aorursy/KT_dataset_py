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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



from sklearn import metrics, preprocessing

from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA, KernelPCA

from tqdm import tqdm



sns.set_style('darkgrid')



pd.options.display.float_format = '{:,.3f}'.format



print(os.listdir("../input"))
%%time

train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_trn = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

test_trn = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
print(train_id.shape, test_id.shape)

print(train_trn.shape, test_trn.shape)
train_id.head()
train_trn.head()
[c for c in train_trn.columns if c not in test_trn.columns]
fc = train_trn['isFraud'].value_counts(normalize=True).to_frame()

fc.plot.bar()

fc.T
# Not all transactions have corresponding identity information.

#len([c for c in train_trn['TransactionID'] if c not in train_id['TransactionID'].values]) #446307



# Not all fraud transactions have corresponding identity information.

fraud_id = train_trn[train_trn['isFraud'] == 1]['TransactionID']

fraud_id_in_trn = [i for i in fraud_id if i in train_id['TransactionID'].values]

print(f'fraud data count:{len(fraud_id)}, and in trn:{len(fraud_id_in_trn)}')
train_id_trn = pd.merge(train_id, train_trn[['isFraud','TransactionID']])

train_id_f0 = train_id_trn[train_id_trn['isFraud'] == 0]

train_id_f1 = train_id_trn[train_id_trn['isFraud'] == 1]

del train_id_trn

print(train_id_f0.shape, train_id_f1.shape)



def plotHistByFraud(col, bins=20, figsize=(8,3)):

    with np.errstate(invalid='ignore'):

        plt.figure(figsize=figsize)

        plt.hist([train_id_f0[col], train_id_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])

        

def plotCategoryRateBar(col, topN=np.nan, figsize=(8,3)):

    a, b = train_id_f0, train_id_f1

    if topN == topN: # isNotNan

        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]

        subA = a.loc[a[col].isin(vals.index.values), col]

        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})

    else:

        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})

    df.sort_values('fraud', ascending=False).plot.bar(figsize=figsize)
plotHistByFraud('id_01')
plotHistByFraud('id_02')
plotHistByFraud('id_07')
plotCategoryRateBar('id_15')
plotCategoryRateBar('id_16')
plotCategoryRateBar('id_17', 10)
plotCategoryRateBar('id_19', 20)

plotHistByFraud('id_19')

print('unique count:', train_id['id_19'].nunique())
import pandas_profiling
pandas_profiling.ProfileReport(train_id)