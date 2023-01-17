import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('/kaggle/input/sf-crime/train.csv')

test_df = pd.read_csv('/kaggle/input/sf-crime/test.csv')



sample_submission = pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv')
train_df.head()
train_df.shape
test_df.head()
test_df.shape
train_df.isnull().sum()
import time

from datetime import datetime



train_df['Dates'] = train_df['Dates'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
train_df['Dates'].describe()
train_df['DayOfWeek'].value_counts()
day_count = train_df['DayOfWeek'].value_counts()
plt.figure(figsize=(12,8))



plt.title('# of Crimes Observed by Day of Week')

plt.xlabel('Day of Week')

plt.ylabel('# of Crimes')



sns.barplot(day_count.index, day_count.values)



plt.show()
train_df['Category'].value_counts()
train_df['Category'].value_counts().shape
cat_count = train_df['Category'].value_counts()
plt.figure(figsize=(16,8))



plt.title('Top 20 # of Crimes Observed by Crime Category')

plt.xlabel('Category')

plt.ylabel('# of Crimes')

plt.xticks(rotation=45, horizontalalignment='right')



sns.barplot(cat_count.index[:20], cat_count.values[:20])



plt.show()
train_df['Descript'].value_counts().shape
desc_count = train_df['Descript'].value_counts()
plt.figure(figsize=(16,8))



plt.title('Top 20 # of Crimes Observed by Crime Description')

plt.xlabel('Descript')

plt.ylabel('# of Crimes')

plt.xticks(rotation=45, horizontalalignment='right')



sns.barplot(desc_count.index[:20], desc_count.values[:20])



plt.show()
train_df['PdDistrict'].value_counts()
pd_count = train_df['PdDistrict'].value_counts()
plt.figure(figsize=(16,8))



plt.title('Top 20 # of Crimes Observed by Police District')

plt.xlabel('PdDistrict')

plt.ylabel('# of Crimes')

plt.xticks(rotation=45, horizontalalignment='right')



sns.barplot(pd_count.index[:20], pd_count.values[:20])



plt.show()
train_df['Resolution'].value_counts()
res_count = train_df['Resolution'].value_counts()
plt.figure(figsize=(16,8))



plt.title('Top 20 # of Crimes Observed by Crime Resolution')

plt.xlabel('Resolution')

plt.ylabel('# of Crimes')

plt.xticks(rotation=45, horizontalalignment='right')



sns.barplot(res_count.index[:20], res_count.values[:20])



plt.show()
train_df['Address'].value_counts()[:50]
add_count = train_df['Address'].value_counts()
plt.figure(figsize=(16,8))



plt.title('Top 20 # of Crimes Observed by Address')

plt.xlabel('Address')

plt.ylabel('# of Crimes')

plt.xticks(rotation=45, horizontalalignment='right')



sns.barplot(add_count.index[:20], add_count.values[:20])



plt.show()
train_df[train_df['Address'] == '200 Block of JONES ST']
train_df[train_df['Address'] == 'TURK ST / JONES ST']
train_df[train_df['Address'] == 'EDDY ST / JONES ST']