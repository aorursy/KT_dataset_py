# import the required libraries

import os, sys

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from sklearn import preprocessing



style.use('seaborn')



%matplotlib inline
DATA_DIR = "/kaggle/input/lish-moa"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))

train_targets_df = pd.read_csv(os.path.join(DATA_DIR, "train_targets_scored.csv"))

test_df = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))

sample_sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print(train_df.shape)

train_df.head()
f"There are {sum('g-' in s for s in train_df.columns)} columns starting with 'g-' that encode the gene expression data and {sum('c-' in s for s in train_df.columns)} starting with 'c-' that encode the cell viability data"
print(train_targets_df.shape)

train_targets_df.head()
test_df.head()
# train_df null check

print(train_df.isnull().values.any())

train_targets_df.isnull().values.any()
# test_df null check

test_df.isnull().values.any()
# sanity check, check if the number of sig_id in train_df is equal to the number of sig_id in the train_targets_df

train_df.sig_id.nunique() == train_targets_df.sig_id.nunique()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))



ax1.set_title("Sample Treatment")

train_df['cp_type'].value_counts().plot.bar(ax=ax1)

ax2.set_title("Treatment Dose")

train_df['cp_dose'].value_counts().plot.bar(ax=ax2)

ax3.set_title("Treatment Duration")

train_df['cp_time'].value_counts().plot.bar(ax=ax3)
plt.figure(figsize=(15,15))

# distributions of the gene expressions

for column in [s for s in train_df.columns if s.startswith('g-')]:

    sns.kdeplot(train_df[column], legend=False)
plt.figure(figsize=(15,15))

# distributions of the cell viability features

for column in [s for s in train_df.columns if s.startswith('c-')]:

    sns.kdeplot(train_df[column], legend=False)
# meta statistics for cell viability

g_stats = train_df[[col for col in train_df if col.startswith('g-')]].describe().T

g_stats[g_stats.columns] = preprocessing.scale(g_stats)



fig, ax = plt.subplots(2, 2, figsize=(15, 7))

fig.suptitle("Meta Statistics for Gene Distribution")



sns.distplot(g_stats['max'], ax=ax[0,0])

sns.distplot(g_stats['mean'], ax=ax[0,1])

sns.distplot(g_stats['min'], ax=ax[1,0])

sns.distplot(g_stats['std'], ax=ax[1,1])
# meta statistics for cell viability

c_stats = train_df[[col for col in train_df.columns if col.startswith('c-')]].describe().T

c_stats[c_stats.columns] = preprocessing.scale(c_stats)



fig, ax = plt.subplots(2, 2, figsize=(15, 7))

fig.suptitle("Meta Statistics for Cell Viablity")



sns.distplot(c_stats['max'], ax=ax[0,0])

sns.distplot(c_stats['mean'], ax=ax[0,1])

sns.distplot(c_stats['50%'], ax=ax[1,0])

sns.distplot(c_stats['std'], ax=ax[1,1])
# frequency distribution of drugs

sns.distplot(train_targets_df.loc[:, train_targets_df.columns != 'sig_id'].sum())
plt.figure(figsize=(15, 7))

plt.title("Drugs with highest MoAs")

train_targets_df.loc[:, train_targets_df.columns != 'sig_id'].sum().sort_values(ascending=False).head(7).plot.barh().invert_yaxis()
plt.figure(figsize=(15, 7))

plt.title("Drugs with lowest MoAs")

train_targets_df.loc[:, train_targets_df.columns != 'sig_id'].sum().sort_values(ascending=False).tail(7).plot.barh().invert_yaxis()
# get the top k endings of the column except for the sig_id column

plt.figure(figsize=(13, 6))

plt.title("Class Name Endings Frequency")

pd.Series([s.split('_')[-1] for s in train_targets_df.columns[1:]]).value_counts().head(10).plot.barh().invert_yaxis()
# correlation matrix for first k columns starting with 'g-'

corr = train_df[[s for s in train_df.columns if s.startswith('g-')][:15]].corr()



plt.figure(figsize=(15,10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), cmap=cmap)
# correlation matrix for first k columns starting with 'c-'

corr = train_df[[s for s in train_df.columns if s.startswith('c-')][:15]].corr()



plt.figure(figsize=(15,10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=np.bool)), cmap=cmap)