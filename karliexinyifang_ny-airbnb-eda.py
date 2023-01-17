# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.shape
df.columns
df.host_id.hist(bins=10)
# how many hosts are there? -37457

len(set(df.host_id))
# How many hosts has more than 10 listings? 2967

df.groupby('host_id').filter(lambda x: len(x)>10).shape
# distribution of number of listings of each host

df.calculated_host_listings_count.hist(bins=100)
#number of host with 1/2/3 listings: 32303, 6658, 2853 sum up: 41814/48895

df[df.calculated_host_listings_count==1].shape[0], df[df.calculated_host_listings_count==2].shape[0], df[df.calculated_host_listings_count==3].shape[0], 
df[df.calculated_host_listings_count>3].calculated_host_listings_count.hist(bins=100)
# number of hosts who have more than 10 (less than 150) listings

df[(df.calculated_host_listings_count>9) & (df.calculated_host_listings_count<150)].calculated_host_listings_count.hist(bins=100)
# The host with most listings

df.groupby('host_id').agg({'id':'count'}).sort_values(by='id',ascending=False).head(10)
# what are the neighbourhoods? (number of listings - Mahattan)

df.neighbourhood_group.value_counts()
# how many hosts are there in each neigh..  number of unique host - Manhattan/Brooklyn

df.groupby('neighbourhood_group').agg({'host_id': 'nunique'}).sort_values(by = 'host_id',ascending=False)
# how many listings are there in each area..  number of listings - Williamsburg

df.groupby(['neighbourhood_group','neighbourhood']).agg({'id': 'nunique'}).sort_values(by = 'id',ascending=False).head(10)
# how many hosts are there in each area..  number of unique host - Williamsburg

df.groupby(['neighbourhood_group','neighbourhood']).agg({'host_id': 'nunique'}).sort_values(by = 'host_id',ascending=False).head(10)
df.room_type.value_counts()
# Might be different in each neighbourhood?

df.groupby(['neighbourhood_group','room_type']).agg({'id':'nunique'}).sort_values(by=['neighbourhood_group','id'],ascending=False)
df.groupby('neighbourhood_group').agg({'price':'mean'}).sort_values(by='price',ascending=False)
# for each room type, where is the most expensive

df.groupby(['room_type','neighbourhood_group']).agg({'price':'mean'}).sort_values(by=['room_type','price'],ascending=False)
# total number of reviews per listing, total number of listings

df.groupby('neighbourhood_group').agg({'number_of_reviews':'sum', 'id':'nunique'}).sort_values(by='number_of_reviews',ascending=False)
# average number of reviews per listing. 

df.groupby('neighbourhood_group').agg({'number_of_reviews':'mean'}).sort_values(by='number_of_reviews',ascending=False)
# process time last_review & creat year column

from datetime import datetime

df['last_review_time'] = df.last_review.apply(lambda x: datetime.strptime(x,'%Y-%m-%d') if type(x)==str else None)

df['year'] = df.last_review_time.apply(lambda x: x.year if 'x'!='' else None)
df.year[0]
min(df.last_review_time), max(df.last_review_time)
# number of listings before 2017: 4379/48000

df[df.last_review_time < '2017-01-01'].shape[0]
df.groupby('year').agg({'id':'count'}).sort_values(by='year',ascending=False)
df_1 = df[df.year>=2015]

df_1.shape
df.shape,df[df.year<2015].shape
df.availability_365.hist(bins=100)
df[df.availability_365>=1].availability_365.hist(bins=100)
df_wo = df[(df.year.isna()==True) & (df.availability_365>0)]

df_wo.shape
df_w = df[(df.year.isna()==True)]

df_w.shape
df_2 = pd.concat([df_1,df_wo])

df_2.shape  ## Bingo!!!
df_2.to_csv('NY_airbnb.csv',index=False)
## Find null values

df.isnull().sum()
# replace null value is reviews_per_month with 0

df.fillna({'reviews_per_month':0},inplace=True)

# examine result

df.reviews_per_month.isnull().sum()
q_min = df.groupby('neighbourhood_group')['price'].quantile(0).to_frame().rename(columns={'price':'min_price'})

q_25 = df.groupby('neighbourhood_group')['price'].quantile(0.25).to_frame().rename(columns={'price':'25%'})

q_50 = df.groupby('neighbourhood_group')['price'].quantile(0.50).to_frame().rename(columns={'price':'50%'})

q_75 = df.groupby('neighbourhood_group')['price'].quantile(0.75).to_frame().rename(columns={'price':'75%'})

q_max = df.groupby('neighbourhood_group')['price'].quantile(1).to_frame() .rename(columns={'price':'max_price'})
quantile = pd.concat([q_min,q_25,q_50,q_75,q_max],axis=1)

quantile

# Manhattan>Brooklun>Queens>Staten Island>Bronx
# Violin plot without extreme value

import seaborn as sns

violin = df[df.price<500]

plot = sns.violinplot(data=violin,x='neighbourhood_group',y='price')

plot.set_title('density and distribution of prices for each neighbourhood_group')
# top 10 neighbourhood:

top_10_nei = df.neighbourhood.value_counts().head(10).index.tolist()
df_top10 = df[df.neighbourhood.isin(top_10_nei)]
listing_count = sns.catplot(x='neighbourhood',hue='neighbourhood_group',col='room_type',data=df_top10,kind='count')

listing_count.set_xticklabels(rotation=90)