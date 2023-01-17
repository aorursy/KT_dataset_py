!pip install fbprophet -q

!pip install dostoevsky -q
import re

import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

from fbprophet import Prophet
FILE = '/kaggle/input/instagram-user/json_data.txt'

f = open(FILE, encoding='utf8')

data = [eval(us) for us in f]

df = json_normalize(data)

df['graphql.shortcode_media.taken_at_timestamp'] = pd.to_datetime(df['graphql.shortcode_media.taken_at_timestamp'],unit='s')

df.head(5)
# retrieve data

comments_parser = []

for i in df['graphql.shortcode_media.edge_media_to_parent_comment.edges']:

    if i:

        for cm in i:

            com = (cm['node'])

            comments_parser.append(com)

comments = json_normalize(comments_parser)

comments['created_at'] = pd.to_datetime(comments['created_at'],unit='s')
comments.head()
comments['owner.is_verified'].value_counts()
# Сhecking for absolute and parse number of comments

comment_df_count = df.set_index('graphql.shortcode_media.taken_at_timestamp').resample('M')['graphql.shortcode_media.edge_media_preview_comment.count'].sum()

comment_count_comment = comments.set_index('created_at').resample('M')['id'].count()

plt.figure(figsize=(15,5))

plt.title('Сhecking for absolute and parse number of comments')

plt.plot(comment_df_count.index, comment_df_count.values)

plt.plot(comment_count_comment.index, comment_count_comment.values)

plt.show()
# Basic Time Series Analysis

comm_data = comments.set_index('created_at').resample('D')['id'].count()

comm_data = pd.DataFrame(comm_data).reset_index()

comm_data.columns = ['ds', 'y']

test_comm = comm_data[:-100]



m = Prophet()

m.fit(test_comm)



future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)



m.plot_components(forecast)

m.plot(forecast)
# Сommentators rating

commentators_top = comments['owner.username'].value_counts().head(10).index

commentators_top = [f'https://www.instagram.com/{x}/?hl=ru' for x in commentators_top]

commentators_top
top_comments = comments['owner.username'].value_counts().head(10)

plt.figure(figsize=(15,5))

plt.title('Сommentators rating')

plt.xticks(rotation=90)

plt.bar(top_comments.index, top_comments.values)

plt.show()
# verified commentators

plt.figure(figsize=(10,10))

comments['owner.is_verified'].value_counts().plot.pie(autopct='%1.1f%%')

plt.legend(['- Plain status','☆Start status'], fontsize=15)

plt.title('Verified commentators')
# Star Commentators

star = comments.loc[comments['owner.is_verified'] == True]

star = [f'https://www.instagram.com/{x}/?hl=ru' for x in star['owner.username'].value_counts().index]

star
# Popular comments (likes)

top_comments = comments.sort_values(['edge_liked_by.count'], ascending = False).head(10)

for text, count in zip(top_comments['text'], top_comments['edge_liked_by.count']):

    print (f'{count} likes: {text[0:100]}...')
# Most of all disputes

disputes_comments = comments.sort_comments = comments.sort_values(['edge_threaded_comments.count'], ascending = False).head(10)

for text, count in zip(disputes_comments['text'], disputes_comments['edge_threaded_comments.count']):

    print (f'{count} reply disputes: {text[0:100]}...')




posting = df.set_index('graphql.shortcode_media.taken_at_timestamp').resample('M')['graphql.shortcode_media.id'].count()

posting_df = pd.DataFrame(posting).reset_index()

posting_df.columns = ['ds', 'y']

sting_confirmed = posting_df[:-30]



m = Prophet()

m.fit(posting_df)

future = m.make_future_dataframe(periods=30)

forecast = m.predict(future)



forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot_components(forecast)
m.plot(forecast)
df.head()
plt.figure(figsize=(10,10))

df['graphql.shortcode_media.__typename'].value_counts().plot.pie(autopct='%1.1f%%')

plt.title('Type content')
df['text'] = [x for x in df['graphql.shortcode_media.edge_media_to_caption.edges']]
for i in df['text']:

    if i:

        print (i[0]['node']['text'])