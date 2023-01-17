# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv')

df.head()
df.info()
df.describe()
df['Subscribers'].value_counts()
missing_val = df['Subscribers'].value_counts().index[0]

df_1 = df[df['Subscribers'] != missing_val]

df_1.info()
df_1['Subscribers'] = pd.to_numeric(df_1['Subscribers'])

df_1.describe()
vid_upload_no_miss = pd.to_numeric(df_1['Video Uploads'], errors = 'coerce')

vid_upload_no_miss
vid_upload_no_miss_fill = vid_upload_no_miss.fillna(value = -1)

vid_upload_no_miss_fill.describe()
df_2 = df_1.drop('Video Uploads', axis = 1)

df_3 = pd.concat([df_2, vid_upload_no_miss_fill], axis = 1)

df_3 = df_3[df_3['Video Uploads'] != -1]

df_3.describe()
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize = (20, 20))

df_3.hist(bins = 50, ax = ax)
df_3.head()
import seaborn as sns

df_subs_and_channels_top_50 = df_3[['Channel name', 'Subscribers']].sort_values('Subscribers', ascending = False)[:50]

fig, ax = plt.subplots(figsize = (12.5, 12.5))

sns.barplot(y = 'Channel name', x = 'Subscribers', data = df_subs_and_channels_top_50, orient = 'h',ax = ax)
df_views_and_channels_top_50 = df_3[['Channel name', 'Video views']].sort_values('Video views', ascending = False)[:50]

fig, ax = plt.subplots(figsize = (12.5, 12.5))

sns.barplot(y = 'Channel name', x = 'Video views', data = df_views_and_channels_top_50, orient = 'h',ax = ax)
df_views_and_channels_top_50 = df_3[['Channel name', 'Video Uploads']].sort_values('Video Uploads', ascending = False)[:50]

fig, ax = plt.subplots(figsize = (12.5, 12.5))

sns.barplot(y = 'Channel name', x = 'Video Uploads', data = df_views_and_channels_top_50, orient = 'h',ax = ax)
import seaborn as sns

sns.countplot(df_3['Grade'])
df_rank_grade = df_3[['Rank', 'Grade']]
l = list(df_rank_grade['Rank'])

rank_list = [i.split(',')[0] + i.split(',')[1] if i[1] == ',' else i[:-2] for i in l]

final_list = [i[:-2] if len(i) == 6 else i for i in rank_list]

int_rank_series = pd.DataFrame(final_list, columns = ['Rank'], dtype = int)

df_rank_grade = df_rank_grade.drop('Rank', axis = 1)

df_rank_grade_new = pd.concat([df_rank_grade, int_rank_series], axis = 1)

df_rank_grade_new.head()
fig, ax = plt.subplots(figsize = (15, 15))

sns.boxplot(x = 'Grade', y = 'Rank', data = df_rank_grade_new, ax = ax)
df_rank_grade_new[df_rank_grade_new['Grade'] == 'A++ ']
fig, ax = plt.subplots(figsize = (15, 15))

sns.boxplot(x = 'Grade', y = 'Subscribers', data = df_3)   
df_3['Subscribers'].plot()

plt.xlabel('Rank of the channel.')

plt.ylabel('Number of Subscribers.')
df_3['Video Uploads'].plot()

plt.xlabel('Rank of the channel.')

plt.ylabel('Number of uploaded videos.')
sns.boxplot(x = 'Grade', y = 'Video Uploads', data = df_3)
df_3['Video views'].plot()

plt.xlabel('Rank')

plt.ylabel('Total number of views')
sns.boxplot(x = 'Grade', y = 'Video views', data = df_3)
sns.scatterplot(x = 'Video views', y = 'Subscribers', data = df_3)
corr_coef = np.corrcoef(x = df_3['Video views'], y = df_3['Subscribers'])

corr_coef
sns.scatterplot(x = 'Video Uploads', y = 'Subscribers', data = df_3)
corr_coef = np.corrcoef(x = df_3['Video Uploads'], y = df_3['Subscribers'])

corr_coef
fig, ax = plt.subplots(figsize = (10, 10))

sns.scatterplot(x = 'Video Uploads', y = 'Video views', size = 'Subscribers', data = df_3, ax = ax)
df_3['Views per Upload'] = df_3['Video views'] / df_3['Video Uploads']

df_3['Views per Upload'].head()
sns.scatterplot(x = 'Views per Upload', y = 'Subscribers', data = df_3)
corr_coef = np.corrcoef(x = df_3['Views per Upload'], y = df_3['Subscribers'])

corr_coef