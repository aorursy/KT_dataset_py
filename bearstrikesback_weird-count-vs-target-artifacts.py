import pandas as pd

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv') 



l3 = np.load('../input/list-of-fake-samples-and-public-private-lb-split/synthetic_samples_indexes.npy')

test = test[~test.index.isin(l3)]



coltouse = [col for col in train.columns.tolist() if col not in ['ID_code', 'target']]

all_df = pd.concat([train, test], sort=False)

for col in tqdm(coltouse):

    all_df[f'scaled_{col}'] = all_df.groupby([col])[col].transform('count')

    

train = all_df[~all_df['target'].isnull()]

test = all_df[all_df['target'].isnull()]
plt.figure(figsize=(10,6))

maxbins = train[train['target'] == 0]['scaled_var_10'].value_counts().shape[0]

plt.title("Distribution of scaled_var_10.")

sns.distplot(train[train['target'] == 0]['scaled_var_10'], color="red", kde=True, bins=maxbins, label='target = 0')

sns.distplot(train[train['target'] == 1]['scaled_var_10'], color="blue", kde=True, bins=maxbins, label='target = 1')

plt.legend(); plt.show()
train[train['target']==0]['scaled_var_10'].value_counts().index, train[train['target']==1]['scaled_var_10'].value_counts().index
train[train['target']==0]['scaled_var_10'].value_counts(normalize=True)
train[train['target']==1]['scaled_var_10'].value_counts(normalize=True)
train[train['target']==0]['scaled_var_10'].value_counts().index.symmetric_difference(train[train['target']==1]['scaled_var_10'].value_counts().index)
all_df[all_df['scaled_var_10']==11]
count = 0

weirdvars = []

idx = []

for col in coltouse:

    if train[train['target']==1][f'scaled_{col}'].value_counts().shape[0] != train[train['target']==0][f'scaled_{col}'].value_counts().shape[0]:

        bins = list(train[train['target']==0][f'scaled_{col}'].value_counts()\

                    .index.symmetric_difference(train[train['target']==1][f'scaled_{col}'].value_counts().index))

        totbins = train[train['target']==0][f'scaled_{col}'].value_counts().shape[0]

        print(f'Variable: {col}, Total bins = {totbins}, Bins with no target=1: {bins}')

        count+=1

        weirdvars.append(col)

        idx.extend(train[train[f'scaled_{col}'].isin(bins)].index)

print(f'Number of vars with artifacts: {count}')

print(f'Number of rows with at least one falsely var in train: {len(np.unique(idx))}')