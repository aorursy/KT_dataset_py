#libraries import

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import numpy as np

import plotly.graph_objects as go

%matplotlib inline
#check of data

NY = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')



NY.head()
NY.info()

print('\nnumber of rows: '+ str(len(NY)))
#deleting columns: id(not importannt), host_name (deleting due to matter of privacy and missing values, moreover we can use host_id instead of host_name)



NY = NY.drop(['id', 'host_name'], axis = 1)



NY.info()
#checking null values

NY.isnull().sum()

#x = NY.isnull().sum().sort_values(ascending = False)[:5]

#y = NY.isnull().sum().sort_values(ascending = False)[:5].index

#plt.figure(figsize=(5,3))

#sns.barplot(x,y, palette = "GnBu_d")
#let notice that if the number_of_reviews is 0 then there last review and reviews_per_month values is null

NY1 = NY[NY.isna().any(axis = 1)]

NY1.head()
NY['reviews_per_month'].mean()
NY['reviews_per_month'].fillna('0', inplace = True)

NY.drop(['last_review'], axis = 1, inplace = True)

NY['name'].fillna('Unknown', inplace = True)
NY.isnull().sum()
# We will skip the first column NAME as I will try to apply some NLP methods on this column later.

# So lets begin with the second column and so on.



top_hosts = NY['host_id'].value_counts()[:15]

top_hosts
#hosts with just one listing

one = NY.loc[NY['calculated_host_listings_count']== 1].iloc[:, 11]

one.sum()
max_l = NY['calculated_host_listings_count'].max()

max_l
NY['neighbourhood_group'].unique()
NY.groupby('neighbourhood_group').agg('count').reset_index()
NY.neighbourhood.unique()
types = NY['room_type'].unique()

#NY['room_type'].loc[NY['room_type']== 'Private room'].count()

room_shares= []

for type in types:

    type_count = int(NY['room_type'].loc[NY['room_type']== type].count())

    share = (type_count/len(NY))*100

    room_shares.append(share)

print(room_shares)
#labels = ['Private room', 'Entire home/apt', 'Shared room']

labels2 = NY['room_type'].unique()

shares = room_shares

explode = (0.1, 0, 0) 

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

texts = plt.pie(shares, explode = explode, colors=colors, labels = labels2, shadow=True, autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.tight_layout()

plt.show()
NY['neighbourhood_group'].value_counts().nlargest(40).plot(kind='bar', figsize = (10,5))

plt.title('neighbourhood')

plt.ylabel('No of bookings per neighbourhood')

plt.xlabel('neighbourhoods')
NY.corr()
plt.figure(figsize = (15,10))

C = NY.corr()

sns.heatmap(C, cmap = 'BrBG',annot = True)
fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(NY['price'], NY['number_of_reviews'])

ax.set_xlabel('price')

ax.set_ylabel('number_of_reviews')

plt.show()
top_hosts = NY['host_id'].value_counts()[0:10].index[0:10]

print(top_hosts)
top = NY[NY['host_id'].isin(list(top_hosts))]

print(top)
price_neighberhood = NY[NY['price'] < 500]

sns.violinplot(data=price_neighberhood, x = 'neighbourhood_group', y = 'price')
sub_brooklyn = NY.loc[NY['neighbourhood_group'] == 'Brooklyn' ]

sub_manhattan = NY.loc[NY['neighbourhood_group'] == 'Manhattan']

sub_queens = NY.loc[NY['neighbourhood_group'] == 'Queens']

sub_staten_island = NY.loc[NY['neighbourhood_group'] == 'Staten Island']

sub_bronx = NY.loc[NY['neighbourhood_group'] == 'Bronx']
price_distribution = go.Figure(data=[go.Histogram(x = sub_brooklyn['price'], name = 'Brooklyn')])



price_distribution.add_trace(go.Histogram(x = sub_manhattan['price'], name = 'Manhatton'))

price_distribution.add_trace(go.Histogram(x = sub_queens['price'], name = 'Queens'))

price_distribution.add_trace(go.Histogram(x = sub_staten_island['price'], name = 'Staten Island'))

price_distribution.add_trace(go.Histogram(x = sub_bronx['price'], name = 'Bronx'))



price_distribution.update_layout(title_text='Price distribution per neighbourhood', xaxis_title_text = 'Price', yaxis_title_text = 'Count', barmode = 'overlay')

price_distribution.update_traces(opacity = 0.5)

price_distribution.show()