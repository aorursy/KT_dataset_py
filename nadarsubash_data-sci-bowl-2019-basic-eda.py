# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

data_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
data_train.head()
dt = data_train[data_train['event_code']==4100]

dt.head()
data_specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

data_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
data_labels.head()
print('---------------Train Data---------------')

print(data_train.info())

print('---------------Test Data---------------')

print(data_test.info())

print('---------------Labels Data---------------')

print(data_labels.info())


print('---------------Train Data---------------')

print(data_train.isna().sum())

print('---------------Test Data---------------')

print(data_test.isna().sum())
print('---------------Train Data---------------')

print(data_train.installation_id.nunique())

print('---------------Test Data---------------')

print(data_test.installation_id.nunique())
data_train.world.value_counts().plot(kind='bar')
grp = data_train.groupby(['world','title'])
grp.size()
grp1 = data_train.query("event_code==4100 or event_code==4110").groupby(['world','title'])
grp1.size()
grp1.size().plot(kind='pie')
grpt = data_train.groupby(['world','type'])

grpt.size()
grpt.size().plot(kind='pie')