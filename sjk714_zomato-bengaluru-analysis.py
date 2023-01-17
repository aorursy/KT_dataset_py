import numpy as np 

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import matplotlib.colors as mcolors

%matplotlib inline
df_main = pd.read_csv("../input/zomato.csv")

df_main.describe()
df_main.info()
df_main.head(1)
df_loc = df_main['location'].value_counts()[:20]

plt.figure(figsize=(20,10))

sns.barplot(x=df_loc,y=df_loc.index)

plt.title('Top 20 locations with highest number of Restaurants.')

plt.xlabel('Count')

plt.ylabel('Restaurant Name')
df_BTM =df_main.loc[df_main['location']=='BTM']

df_BTM_REST= df_BTM['rest_type'].value_counts()



fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(121)



sns.barplot(x=df_BTM_REST, y= df_BTM_REST.index,ax=ax1)

plt.title('Count of restaurant types in BTM')

plt.xlabel('Count')

plt.ylabel('Restaurant Name')



plt.figure(figsize=(20,15))

df_BTM_REST1 = df_BTM_REST[:10]

labels = df_BTM_REST1.index

explode = (0.1, 0,0,0,0,0,0,0,0,0)  

plt.pie(df_BTM_REST1.values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('top 10 restaurant types in BTM')



print("Quick bites are {} % of all the Restaurant types".format((df_BTM_REST.values[0]/df_BTM_REST.sum())*100))
df_RATE_BTM=df_BTM[['rate','rest_type','online_order','votes','book_table','approx_cost(for two people)','listed_in(type)','listed_in(city)','cuisines','reviews_list','listed_in(type)']].dropna()

df_RATE_BTM['rate']=df_RATE_BTM['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)

df_RATE_BTM['approx_cost(for two people)']=df_RATE_BTM['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))



df_rating = df_BTM['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()





f, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True)

sns.despine(left=True)





sns.distplot(df_rating,bins= 20,ax = axes[0]).set_title('Rating distribution in BTM Region')



plt.xlabel('Rating')



df_grp= df_RATE_BTM.groupby(by= 'rest_type').agg('mean').sort_values(by='votes', ascending=False)





sns.distplot(df_grp['rate'],bins= 20,ax = axes[1]).set_title('Average Rating distribution in BTM Region')



df_grp.reset_index(inplace=True)
plt.figure(figsize=(20,10))

sns.barplot(x=df_grp['votes'], y= df_grp['rest_type']).set_title('Average Votes distribution in BTM Region')





df_grp1= df_RATE_BTM.groupby(by='rest_type').agg('mean').sort_values(by='rate', ascending=False)

plt.figure(figsize=(20,10))

sns.barplot(y= df_grp1.index, x=df_grp1['rate']).set_title('Average Rating distributed in BTM Region')

plt.xlim(2.5,5)





df_grp2 =  df_RATE_BTM.groupby(by= 'rest_type').agg('mean').sort_values(by='approx_cost(for two people)', ascending=False)



plt.figure(figsize=(20,10))

sns.barplot(y= df_grp2.index, x=df_grp2['approx_cost(for two people)']).set_title('Average Cost for 2 distributed in BTM Region')
df_Count_CasualDinning =df_main.loc[df_main['rest_type'] =='Casual Dining, Bar'].groupby(by='location').agg('count').sort_values(by='rest_type')

plt.figure(figsize=(10,10))

sns.barplot(x=df_Count_CasualDinning['rest_type'], y= df_Count_CasualDinning.index).set_title("Count of Casual Dining, Bar in Bengaluru")

print('There are about {} number of Casual Dining, Bar in Bengaluru.'.format(df_Count_CasualDinning['rest_type'].sum()))
df_count_casual= df_main.loc[df_main['rest_type'] =='Casual Dining, Microbrewery'].groupby(by='location').agg('count').sort_values(by='rest_type')
sns.barplot(x=df_count_casual['name'],y=df_count_casual.index).set_title('Number of Casual Dining, Microbrewery in Bengaluru ')

plt.xlabel('Count')
df_count_online1=df_main.groupby(by='online_order').agg('count')

df_count_online1.reset_index(inplace=True)



ax1 = plt.subplot2grid((2,2),(0,0))

plt.pie(df_count_online1['url'], labels=df_count_online1['online_order'], autopct='%1.1f%%', shadow=True)



plt.title('online orders?')



df_count_online=df_main.groupby(by='book_table').agg('count')

df_count_online.reset_index(inplace=True)



ax1 = plt.subplot2grid((2,2), (0, 1))

plt.pie(df_count_online['url'], labels=df_count_online['book_table'], autopct='%1.1f%%', shadow=True)

plt.title('Book Table ?')
df_count_online1=df_RATE_BTM.groupby(by='online_order').agg('mean')

df_count_online1.reset_index(inplace=True)

fig, axarr = plt.subplots(1, 2, figsize=(12, 8))



sns.barplot(x=df_count_online1['online_order'], y=df_count_online1['rate'], ax= axarr[0]).set_title('Average rating for book table Restaurants')



sns.barplot(x=df_count_online1['online_order'], y=df_count_online1['votes'],ax= axarr[1]).set_title('Average votes for book table Restaurants')
df_count_online1=df_RATE_BTM.groupby(by='book_table').agg('mean')

df_count_online1.reset_index(inplace=True)



fig, axarr = plt.subplots(1, 2, figsize=(12, 8))



sns.barplot(x=df_count_online1['book_table'], y=df_count_online1['rate'], ax= axarr[0]).set_title('Average rating for book table Restaurants')



sns.barplot(x=df_count_online1['book_table'], y=df_count_online1['votes'],ax= axarr[1]).set_title('Average votes for book table Restaurants')
df_cus= df_RATE_BTM.groupby(by='cuisines').agg('mean').sort_values('rate', ascending = False)[:20]

df_cus.reset_index(inplace=True)
plt.figure(figsize=(20,10))

sns.barplot(x=df_cus['rate'],y=df_cus['cuisines']).set_title('Average Rating on Cuisines')

plt.xlim(3.5,5)
df_cus= df_RATE_BTM.groupby(by='cuisines').agg('mean').sort_values('votes', ascending = False)[:20]

df_cus.reset_index(inplace=True)



plt.figure(figsize=(20,10))

sns.barplot(x=df_cus['votes'],y=df_cus['cuisines']).set_title('Average Votes on Cuisines')



sns.lmplot(x='rate',y='approx_cost(for two people)',data= df_cus)

plt.ylabel('Cost for 2')



sns.lmplot(x='votes',y='approx_cost(for two people)',data= df_cus)

plt.ylabel('Cost for 2')
sns.lmplot(x='rate',y='approx_cost(for two people)',data= df_RATE_BTM)

plt.ylabel('Cost for 2')
sns.lmplot(x='votes',y='approx_cost(for two people)',data= df_RATE_BTM)

plt.ylabel('Cost for 2')


sns.jointplot(y='votes', x='rate', data=df_RATE_BTM, kind='hex',gridsize=20)



sns.jointplot(y='votes', x='rate', data=df_RATE_BTM)

sns.jointplot(y='votes', x='approx_cost(for two people)', data=df_RATE_BTM, kind='hex',gridsize=20)



sns.jointplot(y='votes', x='approx_cost(for two people)', data=df_RATE_BTM)
sns.jointplot(y='rate', x='approx_cost(for two people)', data=df_RATE_BTM, kind='hex',gridsize=20)



sns.jointplot(y='rate', x='approx_cost(for two people)', data=df_RATE_BTM)
import plotly_express as px

px.scatter(df_RATE_BTM, x="rate", y="votes",color='approx_cost(for two people)', marginal_y="violin",

           marginal_x="box", trendline="ols")
tom =df_main['reviews_list'].apply(lambda x: x.replace('RATED\\n',''))
tom = tom.apply(lambda x: x.replace('\'',""))
tom = tom.apply(lambda x: x.replace('\\n',""))
tom[0].split(')')