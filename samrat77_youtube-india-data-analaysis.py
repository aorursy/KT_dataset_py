# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns',10)

pd.set_option('display.max_rows',10)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



data = pd.read_csv('../input/youtube-new/INvideos.csv')

category_data = pd.read_json('../input/youtube-new/IN_category_id.json')
df = pd.DataFrame(data)

print(df.columns)
print(df.info())
print(df.isnull().any())
df = df.drop(['description'], axis = 1)

print(df.columns)
df['trending_date'] = pd.to_datetime(df['trending_date'], format = '%y.%d.%m').dt.date

print(df['trending_date'])
#print(df['publish_time'])

publish_time = pd.to_datetime(df['publish_time'], format = '%Y-%m-%dT%H:%M:%S.%fZ')

print(df['publish_time'])

df['publish_date'] = publish_time.dt.date

df['publish_time'] = publish_time.dt.time

df['publish_hour'] = publish_time.dt.hour



print(df.columns)
print(df[['publish_date','publish_time','publish_hour']])
print(category_data['items'][0])
categories = {category['id']: category['snippet']['title'] for category in category_data['items']}

df.insert(4, 'category', df['category_id'].astype(str).map(categories))

print(df.columns)
df['like_Perc'] = (df['likes'] / (df['likes'] + df['dislikes']))

print(df['like_Perc'])
df['dislike_Perc'] = 1 - df['like_Perc']

print(df['dislike_Perc'])
df_last = df.drop_duplicates(subset = ['video_id'] , keep = 'last' , inplace = False) 

df_first = df.drop_duplicates(subset = ['video_id'] , keep = 'first' , inplace = False)
print('Total number of videos in original datasets are :', (df.shape[0]))

print('Total number of videos in dataset df_last : ', (df_last.shape[0]))

print('Total number of videos in dataset df_first: ', (df_first.shape[0]))
df['Difference_in_days'] = (df['trending_date'] - df['publish_date'])/np.timedelta64(1 , 'D')

df['Difference_in_days'] = df['Difference_in_days'].astype('int')

print(df['Difference_in_days'])
print(df.isnull().sum())
print(df['category'].unique())
df.loc[(df['Difference_in_days'] < 1), 'Difference_in_days'] = 1

df['views_per_day'] = df['views'].astype('int')/df['Difference_in_days']

print(df.head(3))
df['category'].fillna('Nonprofits & Activism', inplace = True)

df[df["category_id"]  == 29]
print(df.isnull().sum())