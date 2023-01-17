# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.

sns.set()
filepath_train = os.path.join('..', 'input', 'train.csv')
filepath_test = os.path.join('..', 'input', 'test.csv')
df_train = pd.read_csv(filepath_train)
df_test = pd.read_csv(filepath_test)
df_train.shape, df_test.shape
df_train.head()
df_train_qid = df_train.copy()

df_train_qid['qid_base_ten'] = df_train_qid['qid'].apply(lambda x : int(x, 16))
df_train_qid.head()
min_qid = df_train_qid['qid_base_ten'].min()
max_qid = df_train_qid['qid_base_ten'].max()
df_train_qid['qid_base_ten_normalized'] = df_train_qid['qid_base_ten'].apply(lambda x : (x - min_qid)/min_qid)
plt.figure(figsize=(18, 8));
plt.scatter(x=df_train_qid['qid_base_ten_normalized'][:100], y=df_train_qid.index[:100]);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index in df_train_qid');
df_test_qid = df_test.copy()

df_test_qid['qid_base_ten'] = df_test_qid['qid'].apply(lambda x : int(x, 16))

df_test_qid['qid_base_ten_normalized'] = df_test_qid['qid_base_ten'].apply(lambda x : (x - min_qid)/min_qid)

plt.figure(figsize=(18, 8));
plt.scatter(x=df_test_qid['qid_base_ten_normalized'][:100], y=df_test_qid.index[:100]);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index in df_test_qid');
df_train_qid.drop('target', axis=1, inplace=True)
df_train_qid['test_or_train'] = 'train'
df_test_qid['test_or_train'] = 'test'
df_qid = pd.concat([df_train_qid, df_test_qid]).sort_values('qid_base_ten_normalized').reset_index()
df_qid.drop('index', axis=1, inplace=True)
df_qid.head()
df_qid_train = df_qid[df_qid['test_or_train']=='train']
df_qid_test = df_qid[df_qid['test_or_train']=='test']

plt.figure(figsize=(18, 8));
plt.scatter(x=df_qid_train['qid_base_ten_normalized'], y=df_qid_train.index, label='Train', s=300);
plt.scatter(x=df_qid_test['qid_base_ten_normalized'], y=df_qid_test.index, label='Test',s=2);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index');
plt.title('qid_base_ten_normalized for train and test datasets')
plt.legend();
plt.figure(figsize=(18, 8));
plt.scatter(x=df_qid_train['qid_base_ten_normalized'][:1500], y=df_qid_train.index[:1500], label='Train');
plt.scatter(x=df_qid_test['qid_base_ten_normalized'][:50], y=df_qid_test.index[:50], label='Test',s=150, marker='d');
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index');
plt.title('qid_base_ten_normalized for the first 1500 train points and 50 test points')
plt.legend();
df_train_qid.head()
df_train_0 = df_train_qid[df_train['target']==0][:5000].drop('test_or_train', axis=1)
df_train_1 = df_train_qid[df_train['target']==1][:5000].drop('test_or_train', axis=1)
df_train_0['qid_distance'] = 0

for i in range(1, df_train_0.shape[0]):
    df_train_0.iloc[i, 4] = df_train_0.iloc[i, 3] - df_train_0.iloc[i-1, 3]
df_train_0.head()
df_train_1['qid_distance'] = 0

for i in range(1, df_train_1.shape[0]):
    df_train_1.iloc[i, 4] = df_train_1.iloc[i, 3] - df_train_1.iloc[i-1, 3]
df_train_1.head()
print('The mean \'qid_distance\' for sincere questions is {} with a standard deviation of {}'.format(round(df_train_0['qid_distance'].mean(), 3), round(df_train_0['qid_distance'].std(), 3)))
print('The mean \'qid_distance\' for insincere questions is {} with a standard deviation of {}'.format(round(df_train_1['qid_distance'].mean(), 3), round(df_train_1['qid_distance'].std(), 3)))
fig, axes = plt.subplots(1, 2, figsize=(18, 8));
sns.distplot(df_train_1['qid_distance'], ax=axes[0], color='red', label='Insincere questions', kde=False);
axes[0].legend();
sns.distplot(df_train_0['qid_distance'], ax=axes[1], label='Sincere questions', kde=False);
axes[1].legend();
plt.figure(figsize=(18, 8));
sns.boxplot(df_train_1['qid_distance'], color='red');
plt.title('Insincere questions');
plt.figure(figsize=(18, 8));
sns.boxplot(df_train_0['qid_distance']);
plt.title('Sincere questions');
df_train_qid.head()
df_train_distance = df_train_qid[:100000].drop(['qid_base_ten', 'test_or_train'], axis=1).copy()
df_train_distance['qid_distance'] = 0
for i in range(1, df_train_distance['qid'].shape[0]):
    df_train_distance.iloc[i, 3] = df_train_distance.iloc[i, 2] - df_train_distance.iloc[i-1, 2]
df_train_distance.head()
df_train_distance_0 = df_train_distance[df_train['target']==0]
df_train_distance_1 = df_train_distance[df_train['target']==1]
ax=plt.gca()
sns.distplot(df_train_distance_0['qid_distance'], ax=ax, color='red', label='Insincere questions');
sns.distplot(df_train_distance_1['qid_distance'], ax=ax, label='Sincere questions');
ax.legend();
sns.boxplot(x=df_train_distance['qid_distance'], y=df_train['target'], orient='h');
