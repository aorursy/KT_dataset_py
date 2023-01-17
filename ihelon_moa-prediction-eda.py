import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn
BASE_PATH = '../input/lish-moa/'
os.listdir(BASE_PATH)
df_test_features = pd.read_csv(os.path.join(BASE_PATH, 'test_features.csv'))

df_train_features = pd.read_csv(os.path.join(BASE_PATH, 'train_features.csv'))

df_train_targets_scored = pd.read_csv(os.path.join(BASE_PATH, 'train_targets_scored.csv'))

df_train_targets_nonscored = pd.read_csv(os.path.join(BASE_PATH, 'train_targets_nonscored.csv'))

df_sample_submission = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
df_train_features
df_test_features
len(set(df_train_features.columns) & set(df_test_features.columns))
print('Number of g-X features: ', df_train_features.columns.str.startswith('g-').sum())

print('Number of c-X features: ', df_train_features.columns.str.startswith('c-').sum())
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.title('TRAIN', fontsize=16)

x = df_train_features['cp_type'].value_counts()

plt.bar(x.index, x.values)

plt.subplot(1, 2, 2)

plt.title('TEST', fontsize=16)

x = df_test_features['cp_type'].value_counts()

plt.bar(x.index, x.values);
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.title('TRAIN', fontsize=16)

x = (df_train_features['cp_time'] // 24).value_counts()

plt.xlabel('days', fontsize=15)

plt.xticks([1, 2, 3])

plt.bar(x.index, x.values)

plt.subplot(1, 2, 2)

plt.title('TEST', fontsize=16)

x = (df_test_features['cp_time'] // 24).value_counts()

plt.xlabel('days', fontsize=15)

plt.xticks([1, 2, 3])

plt.bar(x.index, x.values);
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.title('TRAIN', fontsize=16)

x = df_train_features['cp_dose'].value_counts()

plt.bar(x.index, x.values)

plt.subplot(1, 2, 2)

plt.title('TEST', fontsize=16)

x = df_test_features['cp_dose'].value_counts()

plt.bar(x.index, x.values);
tmp_df = df_train_features.loc[:, ['g-0', 'g-1', 'g-2', 'c-97', 'c-98', 'c-99']]



plt.figure(figsize=(8, 8))

sn.heatmap(tmp_df.corr(), annot=True)

plt.show()
df_train_targets_scored.sample(5)
df_sample_submission.sample(5)
len(set(df_train_targets_scored.columns) & set(df_sample_submission.columns))
x = df_train_targets_scored.drop(['sig_id'], axis=1).sum(axis=0).sort_values()[-30:]

plt.figure(figsize=(16, 10))

plt.yticks(fontsize=15)

plt.barh(x.index, x.values);
x = df_train_targets_nonscored.drop(['sig_id'], axis=1).sum(axis=0).sort_values()[-30:]

plt.figure(figsize=(16, 10))

plt.yticks(fontsize=15)

plt.barh(x.index, x.values);
x = df_train_targets_scored.drop(['sig_id'], axis=1).sum(axis=1).value_counts(sort=False)

plt.figure(figsize=(16, 5))

plt.bar(x.index, x.values);
x = df_train_targets_nonscored.drop(['sig_id'], axis=1).sum(axis=1).value_counts(sort=False)

plt.figure(figsize=(16, 5))

plt.bar(x.index, x.values);
plt.figure(figsize=(16, 8))

plt.imshow(df_train_targets_scored.sample(1000).iloc[:, 1:].values.T)

plt.ylabel('Target features', fontsize=15)

plt.xlabel('Rows', fontsize=15)

plt.title('Active target columns for samples', fontsize=16);
plt.figure(figsize=(16, 8))

plt.imshow(df_train_features.iloc[:200, df_train_features.columns.str.startswith('g-')])

plt.colorbar();
plt.figure(figsize=(16, 8))

plt.imshow(df_train_features.iloc[:200, df_train_features.columns.str.startswith('c-')])

plt.colorbar();
plt.figure(figsize=(16, 8))

plt.imshow(df_train_features.iloc[:500, df_train_features.columns.str.startswith('c-') | df_train_features.columns.str.startswith('g-')])

# plt.colorbar();
plt.figure(figsize=(16, 16))

cols = [

    'g-1', 'g-2', 'g-3', 'g-4',

    'g-5', 'g-6', 'g-7', 'g-8',

    'g-10', 'g-13', 'g-16', 'g-20', 

    'g-70', 'g-80', 'g-90', 'g-100']

for i, col in enumerate(cols):

    plt.subplot(4, 4, i + 1)

    plt.hist(df_train_features.loc[:, col], bins=100, alpha=1);

    plt.title(col)
plt.figure(figsize=(16, 16))

cols = [

    'c-1', 'c-2', 'c-3', 'c-4',

    'c-5', 'c-6', 'c-7', 'c-8',

    'c-9', 'c-10', 'c-11', 'c-12', 

    'c-13', 'c-14', 'c-15', 'c-16']

for i, col in enumerate(cols):

    plt.subplot(4, 4, i + 1)

    plt.hist(df_train_features.loc[:, col], bins=100, alpha=1);

    plt.title(col)
df_train_targets_scored.sum()[1:].sort_values()
plt.figure(figsize=(16, 8))

cols = ['g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8']

for i, col in enumerate(cols, 1):

    plt.subplot(2, 4, i)

    tmp_normal = df_train_features[df_train_targets_scored['nfkb_inhibitor'] == 0]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='false nfkb_inhibitor')

    tmp_normal = df_train_features[df_train_targets_scored['nfkb_inhibitor'] == 1]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='true nfkb_inhibitor')

    plt.title(col)

    plt.legend();
plt.figure(figsize=(16, 8))

cols = ['g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8']

for i, col in enumerate(cols, 1):

    plt.subplot(2, 4, i)

    tmp_normal = df_train_features[df_train_targets_scored['proteasome_inhibitor'] == 0]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='false proteasome_inhibitor')

    tmp_normal = df_train_features[df_train_targets_scored['proteasome_inhibitor'] == 1]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='true proteasome_inhibitor')

    plt.title(col)

    plt.legend();
plt.figure(figsize=(16, 8))

cols = ['g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8']

for i, col in enumerate(cols, 1):

    plt.subplot(2, 4, i)

    tmp_normal = df_train_features[df_train_targets_scored['cyclooxygenase_inhibitor'] == 0]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='false cyclooxygenase_inhibitor')

    tmp_normal = df_train_features[df_train_targets_scored['cyclooxygenase_inhibitor'] == 1]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='true cyclooxygenase_inhibitor')

    plt.title(col)

    plt.legend();
plt.figure(figsize=(16, 8))

cols = ['g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8']

for i, col in enumerate(cols, 1):

    plt.subplot(2, 4, i)

    tmp_normal = df_train_features[df_train_targets_scored['dopamine_receptor_antagonist'] == 0]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='false dopamine_receptor_antagonist')

    tmp_normal = df_train_features[df_train_targets_scored['dopamine_receptor_antagonist'] == 1]

    plt.hist(tmp_normal.loc[:, col], bins=50, density=True, alpha=0.5, label='true dopamine_receptor_antagonist')

    plt.title(col)

    plt.legend();