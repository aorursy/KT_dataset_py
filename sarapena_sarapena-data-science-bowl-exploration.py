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
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
train.head()
train.shape
train.type.value_counts()
train[train.type == 'Assessment'][['installation_id']].drop_duplicates()
train.installation_id.value_counts()
keep_id= train[train.type=='Assessment'][['installation_id']].drop_duplicates()
train = pd.merge(train, keep_id, on='installation_id', how='inner')

train.shape
train.head()
train.game_session.value_counts()
train.event_code.value_counts()
train.event_code.hist()
train.head()
train.type.value_counts()
train.installation_id.value_counts()[train.installation_id.value_counts()==1].index[0]
train[train.installation_id == train.installation_id.value_counts()[train.installation_id.value_counts()==1].index[0]]
train[train.index == 3416428]['event_data'][3416428]
train[train.game_time == 0][train.type == 'Assessment']
train[train.type =='Assessment'].game_time.value_counts(bins = 10)
train.game_time.describe()
train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
train_labels.shape
train.shape
train_labels.columns
train_labels.head()
train[train.type == 'Assessment']['installation_id'].value_counts()
train_labels.head()
train_labels.installation_id.value_counts()
train_labels[train_labels.installation_id == '08987c08'].title.value_counts()
train_labels[train_labels.installation_id == '08987c08'].accuracy_group.value_counts()
train_labels[train_labels.installation_id == '08987c08'][train_labels.accuracy_group == 1]