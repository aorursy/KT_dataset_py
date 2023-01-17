import pandas as pd

import json

from textblob import TextBlob

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
f = open("../input/US_category_id.json")

data = f.read()

f.close()

out = json.loads(data)

categories = pd.DataFrame([

    {'category_id':x['id'],

     'category':x['snippet']['title'],

     'assignable':x['snippet']['assignable']

    } for x in out['items']

])
categories.head()

categories['category_id'] = categories['category_id'].astype(int)
USvideos = pd.read_csv('../input/USvideos.csv',error_bad_lines=False)

UScomments = pd.read_csv('../input/UScomments.csv',error_bad_lines=False)

USvideos['category_id'] = USvideos['category_id'].astype(int)
USvideos.head()
USvideos['date'] =  pd.to_datetime(USvideos.date, format="%d.%m")
USvideos=USvideos.merge(categories, on='category_id')

#Merged the categories
#Using Textblob library to calculate sentiments

def sentiment_calc(text):

    try:

        return TextBlob(text).sentiment.polarity

    except:

        return None
UScomments['polarity']=UScomments['comment_text'].apply(sentiment_calc)
def pol2sent(pol):

    if pol<=-.5:

        return "Very negative"

    if pol>-.5 and pol<0:

        return "Negative"

    if pol == 0:

        return "Neutral"

    if pol > 0 and pol<0.5:

        return "Positive"

    if pol>=0.5:

        return "Very positive"

    
def grouped_weighted_avg(values, weights, by):

   return (values * weights).groupby(by).sum() / weights.groupby(by).sum()
Videosentiment = UScomments.groupby("video_id").mean()

Videosentiment['video_id'] = Videosentiment.index

#print(Videosentiment.columns)

Videosentiment.head()
USvideos=USvideos.merge(Videosentiment, on="video_id")
USvideos['videosentiment']=USvideos['polarity'].apply(pol2sent)
Videosentiment['sentiment']=Videosentiment['polarity'].apply(pol2sent)

Videosentiment.to_csv('video_comments_aggregated_sentiment.csv')

#Saving sentiment CSV for submission
USvideos.head()
USVideo_Last = USvideos.sort_values('likes', ascending=False).drop_duplicates('video_id').sort_index()

USVideo_Last.groupby('category').likes.mean().plot(kind="bar")

print(USVideo_Last.groupby('category').likes.mean())

plt.title("Average Likes per Category")

plt.show()
USVideo_Last.groupby('category').likes.sum().plot(kind="bar")

print(USVideo_Last.groupby('category').likes.sum())

plt.title("Total Likes per Category")

plt.show()
USVideo_Last.groupby('category').dislikes.mean().plot(kind="bar")

print(USVideo_Last.groupby('category').dislikes.mean())

plt.title("Average Dislikes per Category")

plt.show()
USVideo_Last.groupby('category').dislikes.sum().plot(kind="bar")

print(USVideo_Last.groupby('category').dislikes.sum())

plt.title("Total Dislikes per Category")

plt.show()
USVideo_Last.groupby('category').comment_total.mean().plot(kind="bar")

print(USVideo_Last.groupby('category').comment_total.mean())

plt.title("Average Number of Comments per Category")

plt.show()
USVideo_Last.groupby('category').comment_total.sum().plot(kind="bar")

print(USVideo_Last.groupby('category').comment_total.sum())

plt.title("Total Comments per Category")

plt.show()
USVideo_Last.groupby('category').polarity.mean().plot(kind="bar")

print(USVideo_Last.groupby('category').polarity.mean())

plt.title("Average Polarity per Category")

plt.show()
USVideo_Last.groupby('category').views.mean().plot(kind="bar")

print(USVideo_Last.groupby('category').views.mean())

plt.title("Average Views per Category")

plt.show()
USVideo_Last.groupby('category').views.sum().plot(kind="bar")

print(USVideo_Last.groupby('category').views.sum())

plt.title("Total Views per Category")

plt.show()
sns.heatmap(USVideo_Last[['views','likes','dislikes','polarity']].corr(), annot=True, fmt=".2f")

plt.show()
print(USVideo_Last.groupby('category')[['likes','comment_total']].corr().ix[0::2,'comment_total'])

USVideo_Last.groupby('category')[['likes','comment_total']].corr().ix[0::2,'comment_total'].plot(kind='bar')

plt.title("Likes vs Comments - Correlation")

plt.show()
print(USVideo_Last.groupby('category')[['likes','views']].corr().ix[0::2,'views'])

USVideo_Last.groupby('category')[['likes','views']].corr().ix[0::2,'views'].plot(kind='bar')

plt.title("Likes vs Views - Correlation")

plt.show()
print(USVideo_Last.groupby('category')[['dislikes','comment_total']].corr().ix[0::2,'comment_total'])

USVideo_Last.groupby('category')[['dislikes','comment_total']].corr().ix[0::2,'comment_total'].plot(kind='bar')

plt.title("Dislikes vs Comments - Correlation")

plt.show()
print(USVideo_Last.groupby('category')[['dislikes','views']].corr().ix[0::2,'views'])

USVideo_Last.groupby('category')[['dislikes','views']].corr().ix[0::2,'views'].plot(kind='bar')

plt.title("Dislikes vs Views - Correlation")

plt.show()
print(USVideo_Last.groupby('category')[['polarity','likes']].corr().ix[0::2,'likes'])

USVideo_Last.groupby('category')[['polarity','likes']].corr().ix[0::2,'likes'].plot(kind='bar')

plt.title("Sentiment vs Likes - Correlation")

plt.show()
print(USVideo_Last.groupby('category')[['polarity','dislikes']].corr().ix[0::2,'dislikes'])

USVideo_Last.groupby('category')[['polarity','dislikes']].corr().ix[0::2,'dislikes'].plot(kind='bar')

plt.title("Sentiment vs Disikes - Correlation")

plt.show()
print(USVideo_Last.groupby('category')[['polarity','views']].corr().ix[0::2,'views'])

USVideo_Last.groupby('category')[['polarity','views']].corr().ix[0::2,'views'].plot(kind='bar')

plt.title("Sentiment vs Views - Correlation")

plt.show()
for i, group in USVideo_Last.groupby('category'):

    sns.heatmap(group[['views','likes','dislikes','polarity']].corr(), annot=True, fmt=".2f")

    plt.title(i)

    plt.show()
VDF = USVideo_Last.sort_values('views', ascending=False).reset_index(drop=True)

VDF = VDF.groupby('category').head(2).reset_index(drop=True)

VDF[['category','title','channel_title','views']].sort_values('category', ascending=True).reset_index(drop=True)
VDF = USVideo_Last.sort_values('likes', ascending=False).reset_index(drop=True)

VDF = VDF.groupby('category').head(2).reset_index(drop=True)

VDF[['category','title','channel_title','likes']].sort_values('category', ascending=True).reset_index(drop=True)
VDF = USVideo_Last.sort_values('dislikes', ascending=False).reset_index(drop=True)

VDF = VDF.groupby('category').head(2).reset_index(drop=True)

VDF[['category','title','channel_title','dislikes']].sort_values('category', ascending=True).reset_index(drop=True)
VDF = USVideo_Last.sort_values('polarity', ascending=False).reset_index(drop=True)

VDF = VDF.groupby('category').head(2).reset_index(drop=True)

VDF[['category','title','channel_title','polarity']].sort_values('category', ascending=True).reset_index(drop=True)
VDF = USVideo_Last.sort_values('polarity', ascending=True).reset_index(drop=True)

VDF = VDF.groupby('category').head(2).reset_index(drop=True)

VDF[['category','title','channel_title','polarity']].sort_values('category', ascending=True).reset_index(drop=True)