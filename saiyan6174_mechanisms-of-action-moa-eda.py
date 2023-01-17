

import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train.head()
test.head()
print('The shape of train',train.shape)

print('The shape of test',test.shape)

df = pd.concat([train, test]) # concat both train and test datasets
import missingno as msno 

msno.matrix(df) 
df.info()
# printing dtpes

x = df.columns.to_series().groupby(df.dtypes).groups

x
print('Number of "g-" features are: ', len([i for i in train.columns if i.startswith('g-')]))

print('Number of "c-" features are: ', len([i for i in train.columns if i.startswith('c-')]))
train_cp_type = train.groupby('cp_type').size()

test_cp_type = test.groupby('cp_type').size()

print('Distribution of cp_type in train:', train_cp_type)

print('Distribution of cp_type in test:', test_cp_type)

concatenated = pd.concat([train.assign(dataset='train'), test.assign(dataset='test')])
fig = plt.figure(figsize=(8,5))

sns.countplot(x='cp_type',hue='dataset',data=concatenated).set_title('distrbution of cp_type for train and test ~ Compound vs Control')
fig = plt.figure(figsize=(8,5))

sns.countplot(x='cp_time',hue='dataset',data=concatenated).set_title('distribution of treatment duration for train and test')
fig = plt.figure(figsize=(8,5))

sns.countplot(x='cp_dose',hue='dataset',data=concatenated).set_title('distribution of cp_dose for train and test')
f, axes = plt.subplots(3, 3, figsize=(20,10), sharex=True)

sns.despine(left=True)

sns.distplot(df['g-1'], color="r",kde=False, ax=axes[0, 0])

sns.distplot(df['g-103'],color="b",kde=False, ax=axes[0, 1])

sns.distplot(df['g-198'], color="r",kde=False, ax=axes[0, 2])

sns.distplot(df['g-333'], color="b", kde=False,ax=axes[1, 0])

sns.distplot(df['g-432'], color="r",kde=False, ax=axes[1, 1])

sns.distplot(df['g-500'], color="b",kde=False, ax=axes[1, 2])

sns.distplot(df['g-615'], color="r",kde=False, ax=axes[2, 0])

sns.distplot(df['g-670'], color="b",kde=False, ax=axes[2, 1])

sns.distplot(df['g-748'], color="r", kde=False, ax=axes[2, 2])



plt.suptitle('Distributions of 9 randomly selected gene expression data')
f, axes = plt.subplots(3, 3, figsize=(20,10), sharex=True)

sns.despine(left=True)

sns.distplot(df['c-1'], color="r",kde=False, ax=axes[0, 0])

sns.distplot(df['c-13'],color="b",kde=False, ax=axes[0, 1])

sns.distplot(df['c-25'], color="r",kde=False, ax=axes[0, 2])

sns.distplot(df['c-37'], color="b",kde=False, ax=axes[1, 0])

sns.distplot(df['c-43'], color="r",kde=False, ax=axes[1, 1])

sns.distplot(df['c-50'], color="b",kde=False, ax=axes[1, 2])

sns.distplot(df['c-75'], color="r", kde=False,ax=axes[2, 0])

sns.distplot(df['c-80'], color="b",kde=False, ax=axes[2, 1])

sns.distplot(df['c-91'], color="r",kde=False, ax=axes[2, 2])



plt.suptitle('Distributions of 9 randomly selected cell viability data')
fig = plt.figure(figsize=(8,5))

sns.countplot(x='cp_dose',hue='cp_time',data=df).set_title('distribution of cp_dose for train and test')
fig = plt.figure(figsize=(8,5))

sns.countplot(x='cp_type',hue='cp_dose',data=df).set_title('distribution of cp_time and cp_dose together')
plt.figure(figsize=(14,8))

sns.heatmap(train.corr()).set_title('Correlation heatmap')
train_g = train[[i for i in train.columns if i.startswith('g-') and i.endswith('0')]] 

plt.figure(figsize=(14,8))

sns.heatmap(train_g.corr()).set_title('Correlation heatmap')
train_g2 = train[[i for i in train.columns if i.startswith('g-') and (i.endswith('00') or i.endswith('50'))]] 



plt.figure(figsize=(14,8))

sns.heatmap(train_g2.corr(),  cmap='YlGnBu').set_title('Correlation heatmap')
train_c = train[[i for i in train.columns if i.startswith('c-')]] 

plt.figure(figsize=(14,8))

sns.heatmap(train_c.corr()).set_title('Correlation heatmap')
train_scored.head()
train_scored.shape
df = train_scored.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

df.columns = ['count', 'sum']

df = df.groupby(['sum'])['count'].count().reset_index()

df
fig = plt.figure(figsize=(8,5))

sns.barplot(x='sum',y='count',data=df)
df2 = train_scored.drop(['sig_id'], axis=1).astype(bool).sum(axis=0).reset_index()

df2.columns = ['target_column', 'sum']
df2_most_MoAs = df2[df2['sum']>=400]

fig = plt.figure(figsize=(8,5))

sns.barplot(x='sum',y='target_column',data=df2_most_MoAs).set_title('the target classes with more MoA instances')
df2_least_MoAs = df2[df2['sum']<10]

fig = plt.figure(figsize=(8,5))

sns.barplot(x='sum',y='target_column',data=df2_least_MoAs).set_title('the target classes with few MoA instances')