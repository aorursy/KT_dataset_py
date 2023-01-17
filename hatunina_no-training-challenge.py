import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.metrics import roc_auc_score



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)
all_df = pd.concat([train, test], sort=False)

bin_df = all_df.copy()

cols = [col for col in train.columns if col not in ['ID_code', 'target']]



for col in tqdm(cols):

    out = pd.cut(bin_df[col], bins=30, labels=False)

    bin_df['{}_bin'.format(col)] = out.values
print(bin_df.shape)

bin_df.head()
def smooth_mean(df, col, target_col, m):

    mean = df[target_col].mean()

    agg = df.groupby(col)[target_col].agg(['count', 'mean'])

    counts = agg['count']

    means = agg['mean']

    smooth = (counts * means + m * mean) / (counts + m)

    return df[col].map(smooth)



def target_encoding(df):

    cols = [col for col in bin_df.columns if 'bin' in col]

    for col in tqdm(cols):

        df['{}_smooth'.format(col)] = smooth_mean(df, col=col, target_col='target', m=30)

        

    return df
encoding_df = target_encoding(bin_df)
print(encoding_df.shape)

encoding_df.head()
smooth_cols = [col for col in encoding_df.columns if 'smooth' in col]

encoding_df[smooth_cols].head()
all_df['pre'] = encoding_df[smooth_cols].mean(axis=1).values

train = all_df[~all_df['target'].isnull()]

test = all_df[all_df['target'].isnull()]
score = roc_auc_score(train.target, train['pre'])

print('{:.5f}'.format(score))
sub = test[['ID_code', 'target', 'pre']].copy()

del sub['target']

sub = sub.rename(columns={'pre': 'target'})

sub.head()
sub.to_csv('submission.csv', index=False)