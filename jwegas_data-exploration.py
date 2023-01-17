# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# load our data sets
act_train = pd.read_csv('../input/act_train.csv')
act_test = pd.read_csv('../input/act_test.csv')
people = pd.read_csv('../input/people.csv')
# look at train set
act_train.head()

# 2023?? what?? data from the future??
act_train['outcome'].value_counts()
# classes is okey. not unbalanced
# look at test set
act_test.head()
# look at people set
people.head()
# convert field 'data' into data type
act_train['date'] = act_train['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d') )

act_test['date'] = act_test['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d') )
# add a couple of new 'date' fields
act_train['date_year'] = act_train['date'].apply(lambda x: x.year)
act_train['date_month'] = act_train['date'].apply(lambda x: x.month)
act_train['date_day'] = act_train['date'].apply(lambda x: x.day)
act_train['date_weekday'] = act_train['date'].apply(lambda x: x.weekday())

act_test['date_year'] = act_test['date'].apply(lambda x: x.year)
act_test['date_month'] = act_test['date'].apply(lambda x: x.month)
act_test['date_day'] = act_test['date'].apply(lambda x: x.day)
act_test['date_weekday'] = act_test['date'].apply(lambda x: x.weekday())
# 'date' field exploration
act_train['date_year'].value_counts()
act_train['date_month'].hist()
# a lot of actions made at winter months
act_train[act_train['outcome']==1]['date_month'].hist()
# a lot of actions made at winter months
act_train[act_train['outcome']==0]['date_month'].hist()
act_train[act_train['outcome']==1]['date_day'].hist(bins=30)
act_train[act_train['outcome']==0]['date_day'].hist(bins=30)
labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
plt.xticks(np.arange(len(labels)), labels)
act_train['date_weekday'].hist(bins=7)
labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
plt.xticks(np.arange(len(labels)), labels)
act_train[act_train['outcome']==0]['date_weekday'].hist(bins=7)
labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
plt.xticks(np.arange(len(labels)), labels)
act_train[act_train['outcome']==1]['date_weekday'].hist(bins=7)
act_train_grouped = act_train[['people_id', 'outcome']].groupby('people_id')['outcome'].mean().reset_index()
act_train_grouped[(act_train_grouped['outcome'] > 0.0) & (act_train_grouped['outcome'] < 1.0)]['outcome'].hist()
act_train_grouped['outcome'] = act_train_grouped['outcome'].apply(lambda x: np.round(x))
act_train_grouped.info()
people.info()
