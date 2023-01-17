# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sub_df = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

print(sub_df.shape)

sub_df.head()
train_df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

print(train_df.shape)

train_df.head()
train_target_df = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

print(train_target_df.shape)

train_target_df.head()
test_df = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

print(test_df.shape)

test_df.head()
arr = train_target_df.sum(axis=0).values[1:]

arr
fig ,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

ax[0].plot(arr)

ax[0].set_xlabel('Prediction Targets')

sns.distplot(arr, ax=ax[1])

plt.tight_layout()
count_nonzero_labels = train_target_df.sum(axis=1)

print(count_nonzero_labels.unique())

count_nonzero_labels.value_counts().plot(kind='bar')
train_df
train_df[['cp_type','cp_time','cp_dose'][0]].value_counts()
train_df[['cp_type','cp_time','cp_dose'][1]].value_counts()
train_df[['cp_type','cp_time','cp_dose'][2]].value_counts()
train_df['g-4'].plot(kind='hist')
sns.distplot(train_df.iloc[:,4:])


def plot_count_nonzero(data,feature,value, ax):

    data = data.loc[train_df[feature] == value]

    count_nonzero_labels = data.sum(axis=1)

    print(count_nonzero_labels.unique())

    count_nonzero_labels.value_counts().plot(kind='bar', ax=ax)



fig ,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

plot_count_nonzero(train_target_df,'cp_type','trt_cp' ,ax[0])

ax[0].set_xlabel('Feature Distribution (trt_cp)')

plot_count_nonzero(train_target_df,'cp_type','ctl_vehicle' ,ax[1])

ax[1].set_xlabel('Feature Distribution (ctl_vehicle)')

plt.tight_layout()
features_df = train_df.loc[train_df['cp_type'] == 'trt_cp']

features_df = features_df.drop('cp_type', axis=1)

targets_df = train_target_df.loc[features_df.index]

features_df.shape, targets_df.shape
def plot_count_nonzero(data, bool_list,ax): 

    data = data.loc[bool_list]

    count_nonzero_labels = data.sum(axis=1)

    print(count_nonzero_labels.unique())

    count_nonzero_labels.value_counts().plot(kind='bar', ax=ax)



fig ,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))



bool_list = (features_df['cp_dose'] == 'D1')

plot_count_nonzero(targets_df,bool_list ,ax[0])

ax[0].set_xlabel('Feature Distribution (D1)')



bool_list = (features_df['cp_dose'] == 'D2')

plot_count_nonzero(targets_df,bool_list ,ax[1])

ax[1].set_xlabel('Feature Distribution (D2)')



plt.tight_layout()
fig ,ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))



bool_list = (features_df['cp_time'] == 24)

plot_count_nonzero(targets_df,bool_list ,ax[0])

ax[0].set_xlabel('Feature Distribution (24)')



bool_list = (features_df['cp_time'] == 48)

plot_count_nonzero(targets_df,bool_list ,ax[1])

ax[1].set_xlabel('Feature Distribution (48)')



bool_list = (features_df['cp_time'] == 72)

plot_count_nonzero(targets_df,bool_list ,ax[2])

ax[2].set_xlabel('Feature Distribution (72)')



plt.tight_layout()
features_df.iloc[1,4:].plot()
features_df.iloc[1,4:].sort_values().plot()
features_df.iloc[1,4:-100].sort_values().plot()
features_df.iloc[1,-100:].sort_values().plot()
sub_df.iloc[:,1:] = arr/23814

bool_list = (test_df.cp_type == 'ctl_vehicle')

sub_df.iloc[bool_list,1:] = 0
sub_df
sub_df.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv')