# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt



# reading and cleaning data
data=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.isna().sum()
data.dropna()
ii=data.index[data.price==0]
data.drop(ii, inplace=True)# there are 11 records for which price is 0
data
sns.scatterplot(x=data.price,y=data.minimum_nights, hue=data.room_type);
sns.scatterplot(x=np.log(data.price),y=np.log(data.minimum_nights), hue=data.room_type);
plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Brooklyn'])
plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Manhattan'])


plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Queens'])

plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Staten Island'])
plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Bronx'])

nb_group=data.groupby('neighbourhood_group',as_index=False).agg({'host_id':'count'})
nb_group.columns=['neighbourhood_group','Count']
sns.catplot(x='neighbourhood_group', col='room_type', data=data, kind='count')
#to get top 3 hoster's 
data.groupby(('host_name','host_id'),as_index=False).agg({'calculated_host_listings_count':'count'}).sort_values('calculated_host_listings_count',ascending=False)[0:3]  
plt.figure(figsize=(11,10))

sns.relplot(y='price',x='host_name',hue='room_type', data=data[data['host_id'].isin([219517861,107434423,30283594])])
plt.figure(figsize=(11,10))

sns.scatterplot(y='availability_365',x='host_name',data=data[data['host_id'].isin([219517861,107434423,30283594])])
plt.figure(figsize=(11,10))

sns.relplot(x='availability_365',y='price',col='host_name', data=data[data['host_id'].isin([219517861,107434423,30283594])])