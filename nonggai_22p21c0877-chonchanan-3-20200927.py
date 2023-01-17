# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.info()
df.isnull().sum()
print('id_no: ', len(df.index.unique()))
print('name_no: ', len(df['name'].unique()))
print('host_id_no: ', len(df['host_id'].unique()))
print('host_name_no: ', len(df['host_name'].unique()))
print('neighbourhood_group: ', df['neighbourhood_group'].unique())
print('neighbourhood_no: ', len(df['neighbourhood'].unique()))
print('room_type: ', df['room_type'].unique())
df['last_review'] = pd.to_datetime(df['last_review'])
df.isnull().sum()
print('last_review_min: ', df['last_review'].min())
print('last_review_max: ', df['last_review'].max())
neighbourhood_group = pd.get_dummies(df['neighbourhood_group'], prefix='neighbourhood_group')
neighbourhood_group.head()
room_type = pd.get_dummies(df['room_type'], prefix='room_type')
room_type.head()
df = pd.concat([df, neighbourhood_group, room_type], axis=1)
print(df.shape)
df.head()
for i in range(len(df.last_review)):
    
    if (df.last_review[i] >= pd.Timestamp("2018-01-01 00:00:00")):
        df.last_review[i] = 1
    else:
        df.last_review[i] = 0

review_counts = df['last_review'].value_counts()

print('new_review: ', review_counts[1])
print('old_review: ', review_counts[0])
pd.set_option('display.max_columns', None)
df.set_index('id', inplace=True)
df.drop(['name','host_id','host_name','neighbourhood_group','neighbourhood','room_type','last_review'], axis=1, inplace=True)
print(df.shape)
df.head()
df.isnull().sum()
df.fillna(0, inplace=True)
df.isnull().sum()
df.info()
df = (df-df.min())/(df.max()-df.min())
df.head()
df.describe()
df.to_csv('df_clean.csv')
df = pd.read_csv('df_clean.csv', index_col='id')
df.head()
df_sample = df.sample(500)

plt.figure(figsize=(16, 12))
dend = shc.dendrogram(shc.linkage(df_sample, method='ward'))
