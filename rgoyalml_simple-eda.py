# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head()
data.drop(['id', 'name', 'host_name', 'last_review'], axis='columns', inplace=True)

data['reviews_per_month'] = data['reviews_per_month'].fillna(value=0.0)

data.isna().sum()
data.head()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes = axes.flat

sns.barplot(x=data['room_type'].value_counts().index, y=data['room_type'].value_counts(), ax=axes[0]);

axes[0].set(xlabel='Room type', ylabel='Number of Listings', title='Number of Listings per room type');

sns.stripplot(x='room_type', y='price', data=data, ax=axes[1]);

axes[1].set(xlabel='Room type', ylabel='Price', title='Prices of Listings per room type');
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes = axes.flat

sns.barplot(x=data['neighbourhood_group'].value_counts().index, y=data['neighbourhood_group'].value_counts(), ax=axes[0]);

axes[0].set(xlabel='Neighbourhood Group', ylabel='Number of Listings', title='Number of listings per neighbourhood group');

sns.stripplot(x='neighbourhood_group', y='price', data=data, ax=axes[1]);

axes[1].set(xlabel='Neighbourhood Group', ylabel='Price', title='Prices of listings per neighbourhood group');

sns.stripplot(x='neighbourhood_group', y='number_of_reviews', data=data, ax=axes[2]);

axes[2].set(xlabel='Neighbourhood Group', ylabel='number_of_reviews', title='Number of Reviews of Listings per neighbourhood');

fig.delaxes(axes[-1])
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes = axes.flat

sns.barplot(x=data['neighbourhood_group'].value_counts().index, y=data['neighbourhood_group'].value_counts(), ax=axes[0]);

axes[0].set(xlabel='Neighbourhood Group', ylabel='Number of Listings', title='Number of Listings per neighbourhood');

sns.stripplot(x='neighbourhood_group', y='reviews_per_month', data=data, ax=axes[1]);

axes[1].set(xlabel='Neighbourhood Group', ylabel='reviews_per_month', title='Reviews per month of Listings per neighbourhood');
highest_listings_per_host = data['host_id'].value_counts()[data['host_id'].value_counts() > 50]

fig = plt.figure(figsize=(10, 7))

ax = sns.barplot(x=highest_listings_per_host.index, y=highest_listings_per_host);

ax.set(xlabel='host_id', ylabel='Number of Listings');

plt.xticks(rotation=90);
# Collect info on number of reviews per host (using host_id as identifier)

hosts_most_reviews = data['number_of_reviews'].groupby(by=data['host_id']).sum().sort_values(ascending=False)

# hosts_most_reviews = hosts_most_reviews[hosts_most_reviews]

hosts_most_reviews = hosts_most_reviews.reset_index()



# Get number of listings of each host_id

s = data.loc[data['host_id'].isin(hosts_most_reviews['host_id'])]

num_listings_hosts_most_reviews = s['host_id'].value_counts()



# Collect into a Dataframe

host_info = pd.DataFrame({

    'host_id': hosts_most_reviews['host_id'].sort_values(),

    'num_listings': list(num_listings_hosts_most_reviews.sort_index()),

    'num_reviews': hosts_most_reviews.sort_values(by='host_id')['number_of_reviews']

})

host_info = host_info.sort_values(by='num_reviews', ascending=False)

host_info = host_info.loc[(host_info['num_reviews'] > 1000) & (host_info['num_listings'] < 300)]





fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

axes = axes.flat

sns.barplot(x='host_id', y='num_reviews', data=host_info, ax=axes[0]);

sns.barplot(x='host_id', y='num_listings', data=host_info, ax=axes[1])

plt.xticks(rotation=90);
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.scatterplot(x='minimum_nights', y='price', data=data, ax=axes.flat[0]);

sns.scatterplot(x='availability_365', y='price', data=data, ax=axes.flat[1]);

sns.scatterplot(x='number_of_reviews', y='price', data=data, ax=axes.flat[2]);

sns.scatterplot(x='reviews_per_month', y='price', data=data, ax=axes.flat[3]);
mean_p = data['price'].groupby(data['neighbourhood']).mean()

median_p = data['price'].groupby(data['neighbourhood']).median()

total_num_reviews = data['number_of_reviews'].groupby(data['neighbourhood']).sum()

mean_num_reviews = data['number_of_reviews'].groupby(data['neighbourhood']).mean()

data_neighbourhood = pd.DataFrame({'neighbourhood': mean_p.index,

                                       'mean_price': mean_p.values,

                                       'median_price': median_p.values,

                                       'total_num_reviews': total_num_reviews.values})

data_neighbourhood.head()
mean_p = data['price'].groupby(data['neighbourhood_group']).mean()

median_p = data['price'].groupby(data['neighbourhood_group']).median()

total_num_reviews = data['number_of_reviews'].groupby(data['neighbourhood_group']).sum()

mean_num_reviews = data['number_of_reviews'].groupby(data['neighbourhood_group']).mean()

avg_reviews_per_month = data['reviews_per_month'].groupby(data['neighbourhood_group']).mean()

data_neighbourhood_group = pd.DataFrame({'neighbourhood_group': mean_p.index,

                                       'mean_price': mean_p.values,

                                       'median_price': median_p.values,

                                       'total_num_reviews': total_num_reviews.values,

                                        'avg_reviews_per_month': avg_reviews_per_month.values})

data_neighbourhood_group
fig, axes = plt.subplots(3, 1, figsize=(10, 20))

sns.barplot(x='neighbourhood_group', y='mean_price', data=data_neighbourhood_group, ax=axes.flat[0]);

sns.barplot(x='neighbourhood_group', y='median_price', data=data_neighbourhood_group, ax=axes.flat[1]);

sns.barplot(x='neighbourhood_group', y='total_num_reviews', data=data_neighbourhood_group, ax=axes.flat[2]);
d_num_reviews = pd.DataFrame(data['number_of_reviews'].groupby(data['minimum_nights']).sum())

d_num_reviews = d_num_reviews.rename(columns={'number_of_reviews': 'total_num_reviews'})

d_reviews_per_month = pd.DataFrame(data['reviews_per_month'].groupby(data['minimum_nights']).sum())

d_reviews_per_month = d_reviews_per_month.rename(columns={'reviews_per_month':'total_reviews_per_month'})

eval_duration_of_stay = d_num_reviews.join(d_reviews_per_month)

subset_eval_duration_of_stay = eval_duration_of_stay.iloc[:15]
fig, axes = plt.subplots(1, 2, figsize=(15, 5));

sns.barplot(x=subset_eval_duration_of_stay.index, y=subset_eval_duration_of_stay.total_num_reviews, ax=axes.flat[0]);

sns.barplot(x=subset_eval_duration_of_stay.index, y=subset_eval_duration_of_stay.total_reviews_per_month, ax=axes.flat[1]);