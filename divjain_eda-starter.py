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
# Libraries

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline 
train=pd.read_csv(os.path.join(dirname,'train.csv'))

test=pd.read_csv(os.path.join(dirname,'test.csv'))



print(train.shape)

print(test.shape)
train.head()
test.head()
#Adding groups to the train dataset

def add_group(data):

    rows_per_group=500000

    groups =[]

    group_no=0

    

    for i in range(0,len(data),rows_per_group):

        groups.extend([group_no]*rows_per_group)

        group_no+=1

    print('Total Groups:',len(set(groups)))

    return groups

    

train['group_no']=train[['time']].apply(add_group)
train.head()
train.open_channels.nunique()
# To get data level summary

train['signal'].describe()
# Preparing Data Level Summary

data_level_info=train['signal'].describe([0,0.25,0.5,0.75,0.98]).reset_index().T

data_level_info.columns=['data_{}_signal'.format(i) for i in data_level_info.iloc[0,:].tolist()]

data_level_info=data_level_info[1:]

data_level_info['key']=0

data_level_info
train['key']=0

train=train.merge(data_level_info,on='key',how='left')

del train['key']

train.head()
# Preparing Group level summary

group_level_info=train.groupby('group_no')['signal'].apply(lambda x:x.describe([0,0.25,0.5,0.75,0.98])).unstack().reset_index()

group_level_info=group_level_info.drop('count',axis=1)

group_level_info.columns=['group_{}_signal'.format(i) if 'group' not in i else i for i in group_level_info.columns ]

group_level_info
train=train.merge(group_level_info,how='left',on='group_no')

train.head()
sample=train[train['group_no']==0]

sample.columns
rows=train['group_no'].max()

cols = 2



fig, axes = plt.subplots(rows, cols,figsize=(20,10))





for row in range(rows):

    sample=train[train['group_no']==row]

    sample[['signal','data_mean_signal','data_min_signal','data_max_signal','data_50%_signal',

                 'group_mean_signal','group_50%_signal','group_98%_signal']].plot(ax=axes[row,0])

    

    sample['open_channels'].value_counts().plot(kind='bar',ax=axes[row,1])

plt.show()