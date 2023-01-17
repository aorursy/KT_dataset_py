# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



file_path = '/kaggle/input'



import os

for dirname, _, filenames in os.walk(file_path):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Image("/kaggle/input/images/brazil_regions.JPG")
amazon_data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = 'latin1')

# amazon_data = pd.read_csv('./amazon.csv', encoding = 'latin1')
amazon_data.head()
amazon_data.isnull().sum()
print('No. of duplicate obs. = {}'.format(amazon_data.duplicated().sum()))
amazon_data.loc[amazon_data.duplicated(), ].head()
amazon_data.loc[(amazon_data['year'] == 2017) & (amazon_data['state'] == 'Alagoas') & (amazon_data['month'] == 'Janeiro'), ]
amazon_data.loc[(amazon_data['year'] == 1998) & (amazon_data['state'] == 'Mato Grosso') & (amazon_data['month'] == 'Janeiro'), ]
amazon_data.drop_duplicates(inplace = True)

amazon_data.reset_index(inplace = True, drop = True)
print('No. of duplicate obs. = {}'.format(amazon_data.duplicated().sum()))
amazon_data.head()
print('No. of unique values in year variable = {}'.format(amazon_data['year'].nunique()))
amazon_data['year'].unique()
print('No. of unique values in state variable = {}'.format(amazon_data['state'].nunique()))
amazon_data['state'].unique()
print('No. of unique values in state variable = {}'.format(amazon_data['month'].nunique()))
amazon_data['month'].unique()
print('No. of obs. with -ve values in number field = {}'.format(amazon_data['number'].lt(0).sum()))
amazon_data['date'] = pd.to_datetime(amazon_data['date'], format = '%Y-%m-%d')
amazon_data.info()
amazon_data.head()
amazon_data['month'].unique()
amazon_data['date'].dt.day.nunique()
print('No. of obs. with "date" year not matching with "year" variable (out of {} obs.) = {}'.\

      format(amazon_data.shape[0], sum(amazon_data['year'] != amazon_data['date'].dt.year)))
amazon_data.drop(columns = 'date', inplace = True)
tmp_df = pd.DataFrame(amazon_data.groupby(['state', 'year'])['state'].count())

tmp_df.rename(index = str, columns = {'state':'count'}, inplace = True)

tmp_df.reset_index(inplace = True)

tmp_df.groupby('state')['year'].count()
tmp_df.groupby('state')['year'].count().count()
tmp_df = pd.DataFrame(amazon_data.groupby(['state', 'year'])['month'].count())

tmp_df.reset_index(inplace = True)

tmp_df.rename(index = str, columns = {'month':'count'}, inplace = True)

tmp_df.loc[tmp_df['count'] < 12, ]
tmp_df.loc[tmp_df['count'] < 12, 'state'].count()
print('No. of obs. with whole values in "number" variable = {}'.format(amazon_data['number'].apply(lambda x : x.is_integer()).sum()))

print('No. of obs. with float values in "number" variable = {}'.format(amazon_data['number'].apply(lambda x : not x.is_integer()).sum()))
amazon_data.loc[amazon_data['number'].apply(lambda x : not x.is_integer()), ].head()
# Round "number" values to zero decimal places.



amazon_data['number'] = amazon_data['number'].round()
print('No. of obs. with whole values in "number" variable = {}'.format(amazon_data['number'].apply(lambda x : x.is_integer()).sum()))

print('No. of obs. with float values in "number" variable = {}'.format(amazon_data['number'].apply(lambda x : not x.is_integer()).sum()))
yearwise_fire_count_df = pd.DataFrame(amazon_data.groupby('year')['number'].sum())

yearwise_fire_count_df.reset_index(inplace = True)

yearwise_fire_count_df.rename(index = str, columns = {'number':'count'}, inplace = True)

yearwise_fire_count_df = yearwise_fire_count_df.sort_values(by = 'count', ascending = False)

yearwise_fire_count_df.reset_index(inplace = True, drop = True)
plt.figure(figsize = (10, 6))

sns.set(style = 'whitegrid')

sns_graph = sns.barplot(x = 'year', y = 'count', data = yearwise_fire_count_df)

plt.xticks(rotation = 45)

plt.xlabel('Year')

plt.ylabel('No. of Forest Fire Complaints')

plt.title('No. of Forest Fire Complaint across 20 Years')

plt.show()
statewise_fire_count_df = pd.DataFrame(amazon_data.groupby('state')['number'].sum())

statewise_fire_count_df.reset_index(inplace = True)

statewise_fire_count_df.rename(index = str, columns = {'number':'count'}, inplace = True)

statewise_fire_count_df = statewise_fire_count_df.sort_values(by = 'count', ascending = False)

statewise_fire_count_df.reset_index(inplace = True, drop = True)
plt.figure(figsize = (10, 6))

sns.set(style = 'whitegrid')

sns_graph = sns.barplot(x = 'count', y = 'state', data = statewise_fire_count_df)

plt.xticks(rotation = 45)

plt.xlabel('State')

plt.ylabel('No. of Forest Fire Complaints')

plt.title('No. of Forest Fire Complaint across states')

plt.show()
statewise_top10_fire_count_df = statewise_fire_count_df.loc[statewise_fire_count_df.index < 10, ]

statewise_top10_fire_count_df
state_monthwise_avg_number_df = pd.DataFrame(amazon_data.groupby(['state', 'month'])['number'].mean().round())

state_monthwise_avg_number_df.rename(index = str, columns = {'number':'avg'}, inplace = True)

state_monthwise_avg_number_df.reset_index(inplace = True)

state_monthwise_avg_number_df = state_monthwise_avg_number_df.sort_values(by = ['state', 'avg'], ascending = [True, False])

state_monthwise_avg_number_df['rank'] = state_monthwise_avg_number_df.groupby('state')['avg'].rank(ascending = False, method = 'first')

state_monthwise_avg_number_df = state_monthwise_avg_number_df.loc[state_monthwise_avg_number_df['rank'] <= 3, :]

# state_monthwise_avg_number_df
plt.figure(figsize = (20, 60))



unique_states = state_monthwise_avg_number_df['state'].unique()



for i in range(len(unique_states)):

    plt.subplot(8, 3, i+1)

    sns.barplot(y = 'avg', x = 'month', data = state_monthwise_avg_number_df.loc[state_monthwise_avg_number_df['state'] == unique_states[i], ], hue = 'month', dodge = False)

    plt.xlabel('State:' + unique_states[i])

    if i < 3:

        plt.ylabel('No. of Forest Fire Complaints')

        plt.title('Forest Fire Complaints')

    

plt.show()
print('Months across years when maximum no. of forest fire complaints were registered (on average) across top-10 states:')

print()

print(state_monthwise_avg_number_df.loc[state_monthwise_avg_number_df['state'].isin(state_monthwise_avg_number_df['state'].unique()), 'month'].unique().tolist())