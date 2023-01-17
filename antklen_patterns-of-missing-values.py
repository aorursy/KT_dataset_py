import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

print(train.shape)

test = pd.read_csv('../input/test.csv')

print(test.shape)

macro = pd.read_csv('../input/macro.csv')

print(macro.shape)
full = pd.concat([train.drop('price_doc', axis=1), test])

print(full.shape)
full['yearmonth'] = full.timestamp.map(lambda x: x[:7])

train['yearmonth'] = train.timestamp.map(lambda x: x[:7])

test['yearmonth'] = test.timestamp.map(lambda x: x[:7])

macro['yearmonth'] = macro.timestamp.map(lambda x: x[:7])
print('features with missing values in train', (train.isnull().sum()>0).sum())

print('features with missing values in test', (test.isnull().sum()>0).sum())
cols_missing = full.columns[full.isnull().sum()>0].tolist()

print(len(cols_missing))
miss_train = train.groupby('yearmonth')[cols_missing].agg(lambda x: x.isnull().mean()).T

miss_test = test.groupby('yearmonth')[cols_missing].agg(lambda x: x.isnull().mean()).T

miss_train.head()
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[3, 1]},

                               sharey=True, figsize=(12,14))

sns.heatmap(miss_train, ax=ax1, cbar=False, vmax=1, vmin=0)

sns.heatmap(miss_test, ax=ax2, vmax=1, vmin=0)
cols_missing_macro = macro.columns[macro.isnull().sum()>0].tolist()

print('features with missing values in test', len(cols_missing_macro))

miss_macro = macro.groupby('yearmonth')[cols_missing_macro].agg(lambda x: x.isnull().mean()).T



plt.figure(figsize=(12,18))

sns.heatmap(miss_macro)