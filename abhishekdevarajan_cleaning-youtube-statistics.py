import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import json
#creating a list of all csv files to make loading them easier

csvs = glob.glob('../input/youtube-new/*.{}'.format('csv'))

csvs
df_list = []

for csv in csvs:

    if csv[21:23] in ['KR', 'MX', 'JP', 'RU']:

        df = pd.read_csv(csv, index_col = 'video_id', engine='python')

    else:

        df = pd.read_csv(csv, index_col = 'video_id')

    df['country'] = csv[21:23] #This gives the 2 letter country code

    df_list.append(df)

    

yt_df = pd.concat(df_list)

yt_df.head()
yt_df.info()

#From the .head() sample and the .info() description, we can see that there are two date/time related columns

#We should change these columns from dtype = object to dtype = datetime
#Now we will add a category column using the json files provided

yt_df['category_id'] = yt_df['category_id'].astype(str)



category_ids = {}



#Some of the json files were missing one category label so I had to use the US version

#Thus we only need to unpack one json file to fill out our dictionary



j = open('../input/youtube-new/US_category_id.json', 'r')

json_data = json.load(j)

for category in json_data['items']:

    category_ids[category['id']] = category['snippet']['title']



yt_df.insert(4,'category', yt_df['category_id'].map(category_ids))

print(yt_df['category'].unique())

yt_df.head()
#Changing the trending date and publish time to datetime

#Let's also split the publish time column into a publish date and a publish time column

#Notice that the format for trending date is Year.day.month

#the format for publish time is Year-month-day (T)Hour:Minute:Second.milisecond(Z)

from datetime import datetime

yt_df['trending_date'] = pd.to_datetime(yt_df['trending_date'], errors = 'coerce',format = '%y.%d.%m')

yt_df['publish_time'] = pd.to_datetime(yt_df['publish_time'], errors = 'coerce', format = '%Y-%m-%dT%H:%M:%S.%fZ')

yt_df.insert(5,'publish_date', yt_df['publish_time'].dt.date)

yt_df['publish_time'] = yt_df['publish_time'].dt.time

yt_df.info()

yt_df[['trending_date', 'publish_date', 'publish_time']].head(10)
#It looks like there are videos where the video id is missing

print(yt_df.reset_index()['video_id'].str.startswith('#').sum())



#Let us remove these rows

yt_df = yt_df.reset_index()

yt_df = (yt_df[~yt_df['video_id'].str.startswith('#')])

yt_df = yt_df.set_index('video_id')

print(yt_df.reset_index()['video_id'].str.startswith('#').sum())
countries = yt_df['country'].unique()

countries
for country in countries:

    cat_counts_df = yt_df[yt_df['country']==country]['category'].value_counts().reset_index()

    plt.figure(figsize=(15,10))

    sns.set_style("darkgrid")

    ax = sns.barplot(y=cat_counts_df['index'],x=cat_counts_df['category'], data=cat_counts_df,orient='h')

    plt.xlabel("Number of Videos")

    plt.ylabel("Category")

    plt.title(f'Trending Videos by Category in {country}')

    plt.show()
#Reset the index to make video ID a separate column again, then filter out all but the first instance of each video

first_trending = yt_df.reset_index().drop_duplicates('video_id', keep = 'first').set_index('video_id')



#Find the difference between the date published and the first trending date

first_diff = first_trending['trending_date'].astype('datetime64[ns]') - first_trending['publish_date'].astype('datetime64[ns]')

first_diff = first_diff.reset_index()

first_diff.columns = ['video_id', 'pub_to_trend_first']



#Store the time from publishing to trending in a dictionary

pub_to_trend_dict = {}

for row in first_diff.itertuples():

    pub_to_trend_dict[str(row[1])] = row[2].days



#Find the most recent instacne of trending

last_trending = yt_df.reset_index().drop_duplicates('video_id', keep = 'last').set_index('video_id')

last_diff = last_trending['trending_date'].astype('datetime64[ns]') - last_trending['publish_date'].astype('datetime64[ns]')

last_diff = last_diff.reset_index()

last_diff.columns = ['video_id', 'pub_to_trend_last']

yt_df = yt_df.reset_index()



pub_to_trend_dict2 = {}

for row in last_diff.itertuples():

    pub_to_trend_dict2[str(row[1])] = row[2].days



#insert columns for the publication-trending time differences

yt_df.insert(4, 'pub_to_trend_last', yt_df['video_id'].map(pub_to_trend_dict2))

yt_df.insert(4, 'pub_to_trend', yt_df['video_id'].map(pub_to_trend_dict))

yt_df.insert(4, 'trending_duration', 0)



yt_df['trending_duration'] = abs(yt_df['pub_to_trend_last'] - yt_df['pub_to_trend']) +1

yt_df.set_index('video_id')[['pub_to_trend', 'trending_duration']].sort_values('trending_duration', ascending = False).head()
yt_df[['category','trending_duration']].head()
trending_time_category = yt_df.groupby(['category', 'trending_duration']).count()['video_id'].unstack().clip(upper = 500)

yt_df = yt_df.set_index('video_id')

plt.figure(figsize=(15,15))

sns.set_style("dark")

sns.heatmap(trending_time_category, cmap = 'cool')

plt.title('Trending Duration by Video Category')

plt.show()
#Lets create a dataframe that tracks the number of times each video trends, grouped by country

trend_count = yt_df.groupby([yt_df.index, 'country']).count()['title'].sort_values(ascending=False).reset_index()

trend_count.head()
#Create a dataframe to calculate the correlation between countries

corr_df = trend_count.pivot(index='video_id', columns = 'country', values = 'title')

corr_df = corr_df.fillna(0).astype(int)

corr_df['total'] = corr_df.sum(axis=1)

corr_df
countries = [c for c in corr_df.columns if c != 'total']



corr_mat = corr_df[countries].corr()



fig = plt.figure(figsize=(15,15))

sns.heatmap(corr_mat, annot = True)

plt.show()
yt_df.info()

#looks like the interactions stats are already integers. Dope
#List of interactions and the number of countries we have

interactions = ['views', 'likes', 'dislikes', 'comment_count']

country_count = len(countries)



#Creating a dataframe of all the info we want to graph

interact_df = yt_df.groupby(['country'])[interactions].sum()

interact_df['unique_counts'] = pd.DataFrame(yt_df.reset_index().groupby(['country'])['video_id'].nunique())

#Diving the total interaction counts by the number of unique trending videos in each country

interact_df = interact_df.apply(lambda x: x/interact_df['unique_counts'])

interact_df = interact_df.drop('unique_counts', axis=1)





for column in interact_df.columns:

    fig = plt.figure(figsize=(15,10))

    sns.barplot(x = interact_df.index, y = column, data = interact_df)

    plt.title('Number of {} by Country'.format(column.capitalize()))

    plt.show()
t_counts_country = pd.DataFrame(trend_count.groupby(['country'])['title'].value_counts())

t_counts_country.columns = ['title_counts']

t_counts_country = t_counts_country.reset_index()



for country in t_counts_country['country'].unique():

    fig = plt.figure(figsize=(10,10))

    sns.lineplot(x='title', y='title_counts', data = t_counts_country[t_counts_country['country']==country])

    plt.title(f'How Many Times Do Videos Trend in {country}')

    plt.xlabel('Instances of Trending')

    plt.ylabel('Number of Videos')

    plt.show()
like_dislike = yt_df.groupby(['category'])[['likes', 'dislikes']].sum()

for col in like_dislike.columns:

    fig = plt.figure(figsize=(10,10))

    sns.barplot(x = like_dislike.index, y = like_dislike[col], data = like_dislike)

    plt.xticks(rotation = 'vertical')

    plt.title(f'Number of {col.capitalize()} by Category')

    plt.show()
like_dislike['ratio'] = like_dislike['likes']/like_dislike['dislikes']

like_dislike = like_dislike.sort_values(by='ratio', ascending=False)

#This makes the bars red if they have fewer likes than dislikes, and green otherwise

clrs = ['red' if (x < 1) else 'green' for x in like_dislike['ratio'].values ]



fig = plt.figure(figsize=(10,10))

sns.barplot(x = like_dislike.index, y = like_dislike['ratio'], data = like_dislike, palette = clrs)

plt.xticks(rotation = 'vertical')

plt.title('Like/Dislike Ratio by Category')

plt.show()
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk import sent_tokenize, word_tokenize

from itertools import chain

import re
#let's create a list of stopwords from all of the avialable languages in nltk

languages = ['english','french','german','russian','spanish']

stopwords_list = [stopwords.words(lang) for lang in languages]

stopwords_list = list(chain(*stopwords_list))
polarities = [12.5359,10.352399999999996,7.096599999999999,9.528299999999998,16.047199999999997,7.9980999999999955,15.507100000000001,5.694699999999998,1.9956000000000005,-6.599499999999999,

6.049099999999998,9.764000000000005,1.4337,12.204700000000004,9.140699999999994,3.0110000000000006,1.85,0.0]



cat_list = yt_df['category'].unique()
tag_sentiments = pd.concat([pd.DataFrame(cat_list),pd.DataFrame(polarities)], axis=1)

tag_sentiments.columns = ['category','polarity']

tag_sentiments = tag_sentiments.sort_values('polarity').reset_index()
fig = plt.figure(figsize=(10,10))

sns.barplot(x='polarity', y='category', data=tag_sentiments, orient='h', palette='coolwarm_r')

plt.xlabel('Categories')

plt.ylabel('Polarity')

plt.show()
year_counts = yt_df['publish_date'].apply(lambda x: x.year).reset_index()

year_counts = year_counts.groupby('publish_date').count()

years_upto_2016 = year_counts[year_counts.index <= 2016]

years_after_2016 = year_counts[year_counts.index >2016]



fig = plt.figure(figsize=(10,10))

sns.barplot(x=years_upto_2016.index, y=years_upto_2016['video_id'], data=years_upto_2016, palette='Pastel2')

plt.xlabel('Year Published')

plt.ylabel('Number of Trending Videos')

plt.show()



sns.barplot(x=years_after_2016.index, y=years_after_2016['video_id'], data=years_after_2016, palette='Pastel2')

plt.xlabel('Year Published')

plt.ylabel('Number of Trending Videos')

plt.show()
channels = yt_df.groupby('channel_title').size().reset_index(name='video_count')

channels = channels.sort_values('video_count', ascending=False)

channels_head = channels.head(20)
fig = plt.figure(figsize=(10,10))

sns.barplot(x=channels_head['video_count'], y=channels_head['channel_title'], data=channels_head, palette='ocean')

plt.xlabel('Number of Trending Videos')

plt.ylabel('Channel Name')

plt.show()
Metrics = yt_df.drop(['pub_to_trend','pub_to_trend_last'], axis=1)



fig = plt.figure(figsize=(10,10))

sns.heatmap(Metrics.corr(), annot=True, cmap='RdBu')

plt.show()