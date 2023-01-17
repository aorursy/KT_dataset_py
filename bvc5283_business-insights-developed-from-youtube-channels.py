import numpy as np

import pandas as pd

import os

import json

import seaborn as sns

import matplotlib.pyplot as plt

import glob

from wordcloud import WordCloud

import nltk

import datetime

# nltk.download()

from nltk.sentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk import sent_tokenize, word_tokenize

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.tokenize import RegexpTokenizer

import re

from math import pi



data=pd.read_csv("../input/newest-trending-videos/Data.csv")

data.info()

data.head(5)
# Overview of nan values

data.isna().sum()
# Drop null values

data = data[pd.notna(data['title'])]

# Pair the category id with category title

categories={}

with open('../input/youtube-new/US_category_id.json', 'r') as f:

    category = json.load(f)

    for i in category['items']:

        categories[int(i['id'])]=i['snippet']['title']        

data['category']=[categories[i] for i in data['categoryId']]
# Reformat the date and time columns

data['publishedAt'] = pd.to_datetime(data['publishedAt'], format='%Y-%m-%d')

data['trending_date']=pd.to_datetime(data['trending_date'],format='%y.%d.%m')

# Drop columns we do not use

data.drop(['thumbnail_link', 'categoryId','channelId','video_id'],axis=1,inplace=True)

# Remove unknown values wihch are shown as 0 in view_count, likes, dislikes, and comment_count

missing=data.loc[(data['view_count'] ==0) | (data['likes'] ==0)|(data['dislikes'] ==0)|(data['comment_count'] ==0)]

data.head()
# Vlsualize the correlation between view_count, likes,dislikes, and comment_count

correlation=data[['view_count','likes','dislikes','comment_count']].corr()

sns.heatmap(correlation,cmap='Blues',annot=True);
# Plot the number of video by category

plt.figure(figsize = (25,9))

sns.countplot(data['category'],order = data['category'].value_counts().index, palette=sns.color_palette("Blues_r",15))

plt.title('Number of Videos by Category',fontsize=20)

plt.ylabel('Count',fontsize=16)

plt.xlabel('Category',fontsize=16);
# The average of four numeric attributes 

general_view=pd.DataFrame(data[['view_count','likes','dislikes','comment_count']].groupby(data['category']).mean())

plt.figure(figsize=(32,20))

plt.subplot(2,2,1)

plt.plot( general_view.index,'view_count' , data=general_view, color='skyblue', linewidth=2)

plt.title('View_count vs Category',fontsize=20)

plt.xticks(rotation=30)

plt.subplot(2,2,2)

plt.plot( general_view.index, 'likes', data=general_view, color='olive', linewidth=2)

plt.title('Likes vs Category',fontsize=20)

plt.xticks(rotation=30)

plt.subplot(2,2,3)

plt.plot( general_view.index, 'dislikes', data=general_view, color='black', linewidth=2)

plt.title('Dislikes vs Category',fontsize=20)

plt.xticks(rotation=30)

plt.subplot(2,2,4)

plt.plot( general_view.index, 'comment_count', data=general_view, color='red', linewidth=2, linestyle='dashed')

plt.title('Comment_count vs Category',fontsize=20)

plt.xticks(rotation=30);




# Plot the distribution of 'view_count','likes','dislikes','comment_count'

view_count=np.log(data['view_count']+1 )

likes=np.log(data['likes']+1)

dislikes=np.log(data['dislikes']+1)

comment=np.log(data['comment_count']+1)

data_count=pd.concat([view_count,likes,dislikes,comment], axis=1)

data_count.index=data['category']

data_count=data_count[(data_count != 0)]

plt.figure(figsize=(32,25))



# distribution of view counts of different categories

plt.subplot(2,2,1)

sns.boxplot(data_count.index,'view_count', data=data_count,order = data['category'].value_counts().index)

plt.xticks(rotation=30,fontsize=12)

plt.xlabel('')

plt.ylabel('')

plt.title("View Count", fontsize=20)



# distribution of likes of different categories

plt.subplot(2,2,2)

sns.boxplot(data_count.index,'likes', data=data_count,order = data['category'].value_counts().index)

plt.xticks(rotation=30,fontsize=12)

plt.xlabel('')

plt.ylabel('')

plt.title("Likes", fontsize=20)



# distribution of dislikes of different categories

plt.subplot(2,2,3)

sns.boxplot(data_count.index,'dislikes', data=data_count,order = data['category'].value_counts().index)

plt.xticks(rotation=30,fontsize=12)

plt.xlabel('')

plt.ylabel('')

plt.title("Dislikes", fontsize=20)



# distribution of comment counts of different categories

plt.subplot(2,2,4)

sns.boxplot(data_count.index,'comment_count', data=data_count,order = data['category'].value_counts().index)

plt.xticks(rotation=30,fontsize=12)

plt.xlabel('')

plt.ylabel('')

plt.title("Comment Count", fontsize=20);



#the distribution of days that videos take to become popular

data['publish_date'] = data['publishedAt'].dt.date

data['publish_time'] = data['publishedAt'].dt.time

data['interval'] = (data['trending_date'].dt.date-data['publish_date']).astype('timedelta64[D]')

#Histgram of distribution of interval

plt.figure(figsize = (25,9))

sns.countplot(data['interval'], color='skyblue')

plt.title('Time Interval',fontsize=20)

plt.xlabel('Interval',fontsize=16)

plt.ylabel('Count',fontsize=16)

# Average time interval between published and trending

df_t = pd.DataFrame(data['interval'].groupby(data['category']).mean()).sort_values(by="interval")

plt.figure(figsize = (25,9))

plt.plot(df_t, color='skyblue', linewidth=2)

plt.title("Average Days to be trending video", fontsize=20)

plt.xlabel('Category',fontsize=16)

plt.ylabel('Average Time Interval',fontsize=16)

plt.show();



# Create evaluation matrix by mean of time interval, number of videos, average view count, standard deviation of view count

Matrix1 = data.pivot_table(index="category", values=["view_count", "interval"], 

                 aggfunc={"interval": np.mean, "view_count":["count", np.std, np.mean]})

Matrix1.columns = ["interval_mean", "video_count", "view_mean", "view_std"]



# Generate socres for each character by descending the rank

for col in Matrix1.columns:

    Matrix1["{}_rank".format(col)] = Matrix1[col].rank(ascending=1)

Matrix1["interval_mean_rank"] = 15-Matrix1["interval_mean_rank"]



# Calculate total score

Matrix1["score"] = Matrix1.iloc[:,-4:].sum(axis=1)

Matrix1.sort_values(by="score", inplace=True, ascending=False)

# Matrix Ranking

Matrix1.drop(['interval_mean','video_count','view_mean','view_std'],axis=1)
# Visualization of evalustion matrix

plt.figure(figsize = (15,12))

plt.scatter(x=Matrix1['video_count_rank'], y=Matrix1['interval_mean_rank'], s=Matrix1['view_mean_rank']*300, c=Matrix1['view_std_rank'], cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)

plt.xlabel("Video count score",fontsize=16)

plt.ylabel("Interval mean score",fontsize=16)

plt.title('Category Score Demostration',fontsize=20)



for i in range(len(Matrix1)):

    plt.annotate(s=Matrix1.index[i], xy=(Matrix1['video_count_rank'][i]-0.5, Matrix1['interval_mean_rank'][i]+0.5));

# Locate Entertainment

Entertainment=data.loc[data['category']=='Entertainment']



# Plot the frequency of Channel Appearance

Matrix2=pd.DataFrame(Entertainment['channelTitle'].groupby(Entertainment['channelTitle']).count())

Matrix2.columns=['Appearance_Frequency']



data['publishedAt'] = pd.to_datetime(data['publishedAt'], format='%Y-%m-%dT%H:%M:%S').map(lambda x: x.date)

data['trending_date']=pd.to_datetime(data['trending_date'],format='%y.%d.%m').astype("str")

# Create Heatmap

def channelHeatmap(category):

    plt.figure(figsize=(15,20))

    pivot_table = data[data.category == category].pivot_table(values='view_count', index='channelTitle', columns='trending_date', aggfunc=lambda x:np.log(x.mean()), fill_value=0)

    sns.heatmap(pivot_table, cmap = "Greens", vmin = 0)

    plt.title('ViewCount of {}'.format(category),fontsize=20)

    plt.xlabel('Tranding Date',fontsize=16)

    plt.ylabel('Frequency',fontsize=16)

    plt.show();

channelHeatmap("Entertainment")



# Entertainment

Entertainment['likes_to_dislikes'] = Entertainment['likes']/Entertainment['dislikes']

Matrix2['likes_to_dislikes']=pd.DataFrame(Entertainment['likes_to_dislikes'].groupby(Entertainment['channelTitle']).mean())
# Plot for Entertainment

plt.figure(figsize = (10,25))

sns.barplot(x=Matrix2['likes_to_dislikes'], y=Matrix2.index, palette=sns.color_palette("BuGn_r",200),order=Matrix2['likes_to_dislikes'].sort_values(ascending=False).index)

plt.xlabel('likes_to_dislikes ratio',fontsize=16)

plt.ylabel('Channel',fontsize=16)

plt.yticks(fontsize=8) 

plt.xticks(fontsize=12)

plt.title('likes_to_dislikes ratio in Entertainment',fontsize=20)

plt.yticks(fontsize=7) 

plt.show();

Entertainment['growth_rate']=Entertainment['view_count']/(Entertainment['interval']+1)

Matrix2['growth_rate']=pd.DataFrame(Entertainment['growth_rate'].groupby(Entertainment['channelTitle']).mean())

plt.figure(figsize=(10,27))

sns.barplot(x=Matrix2['growth_rate'],y=Matrix2.index,order=Matrix2['growth_rate'].sort_values(ascending=False).index, palette=sns.color_palette("Blues_r",200))

plt.yticks(fontsize=8) 

plt.xticks(fontsize=12)

plt.xlabel('Growth Rate',fontsize=16)

plt.ylabel('Channels',fontsize=16)

plt.title("Growth Rate in Entertainment", fontsize=20);



# Sentiment Analysis

category_list = Entertainment['channelTitle'].unique()

# Collect all the related stopwords.

en_stopwords = list(stopwords.words('english'))

polarities = list()

MAX_N=10000



for i in category_list:

    tags_word = Entertainment[Entertainment['channelTitle']==i]['description'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words

    tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)

    word_tokens = word_tokenize(tags_word)

    filtered_sentence = [w for w in word_tokens if not w in en_stopwords]

    without_single_chr = [word for word in filtered_sentence if len(word) > 2]



# Remove numbers

    cleaned_data_title = [word for word in without_single_chr if not word.isdigit()]      

    

# Calculate frequency distribution

    word_dist = nltk.FreqDist(cleaned_data_title)

    hnhk = pd.DataFrame(word_dist.most_common(MAX_N),

                    columns=['Word', 'Frequency'])



    compound = .0

    for word in hnhk['Word'].head(MAX_N):

        compound += SentimentIntensityAnalyzer().polarity_scores(word)['compound']



    polarities.append(compound)
# Plot of Sentiment

category_list = pd.DataFrame(category_list)

polarities = pd.DataFrame(polarities)

tags_sentiment = pd.concat([category_list,polarities],axis=1)

tags_sentiment.columns = ['channelTitle','polarity']

tags_sentiment.set_index('channelTitle',inplace=True)

tags_sentiment=tags_sentiment.sort_values('polarity')

plt.figure(figsize=(10,25))

ax = sns.barplot(x=tags_sentiment['polarity'],y=tags_sentiment.index, data=tags_sentiment)

plt.xlabel("Polarity",fontsize=16)

plt.ylabel("Entertainment",fontsize=16)

plt.yticks(fontsize=8) 

plt.xticks(fontsize=12)

plt.title("Sentiment Polarity of Entertainment Videos", fontsize=20);
# Matrix for Entainment

Matrix2['Frequency'] = Matrix2['Appearance_Frequency'].rank(ascending=1)

Matrix2['Growth']=Matrix2['growth_rate'].rank(ascending=1)

Matrix2['Ratio']=Matrix2['likes_to_dislikes'].rank(ascending=1)

Matrix2=Matrix2.join(tags_sentiment, how='left')

Matrix2['Sentiment']=Matrix2['polarity'].rank(ascending=1)





Matrix2.drop(['Appearance_Frequency', 'growth_rate','likes_to_dislikes','polarity'],axis=1,inplace=True)

Matrix2['Score']=Matrix2.sum(axis=1)

Matrix2.sort_values('Score',ascending=False,inplace=True)

Matrix2.head(10)



# function of plotting spidermap

def spidermap(Channel_Name):

    Channel=Matrix2.loc[Matrix2.index==Channel_Name]

    Spider = pd.DataFrame({

    'group': [Channel_Name],

    'Frequency': Channel['Frequency'].values,

    'Growth': Channel['Growth'].values,

    'Ratio': Channel['Ratio'].values,

    'Sentiment': Channel['Sentiment'].values,

    })

    N=4

    categories=list(Spider)[1:]

    plt.figure(figsize=(8,8))

    values=Spider.loc[0].drop('group').values.flatten().tolist()

    values += values[:1]

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]

    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], categories, color='grey', size=10)

    ax.set_rlabel_position(0)

    plt.yticks([50,100,150,200], ["50","100","150"], color="grey", size=13)

    plt.ylim(0,200)

    ax.plot(angles, values, linewidth=1, linestyle='solid')

    ax.fill(angles, values, 'b', alpha=0.1)

    plt.title(Channel_Name)

    plt.show()
# function of creating word cloud

def word_cloud(Channel_Name):

    Channel=data.loc[data['channelTitle']==Channel_Name]

    tags_word = data.loc[data['channelTitle']==Channel_Name]['description'].to_string().lower()

    word_tokens = word_tokenize(tags_word)

    cloud = WordCloud(background_color = 'black', colormap="tab20b_r",max_words=2000, max_font_size=40, random_state=42)

    cloud.generate(' '.join(word_tokens))

    

    plt.figure(figsize = (10,10),facecolor = None)

    plt.imshow(cloud)

    plt.axis('off')

    plt.title("\nWord cloud for {}\n".format(Channel_Name), fontsize=20)

    filtered_sentence = [w for w in word_tokens if not w in en_stopwords]

    without_single_chr = [word for word in filtered_sentence if len(word) > 2]

    Channel_Name = [word for word in without_single_chr if not word.isdigit()]
# scores of rank 1 channel

Matrix2.head(1)

channel_view=pd.DataFrame(data[['view_count','likes','dislikes','comment_count']].groupby(data['channelTitle']).mean())

print(channel_view.loc[channel_view.index=='Binging with Babish'])

spidermap('Binging with Babish')

word_cloud('Binging with Babish')



print(channel_view.loc[channel_view.index=='MrBeast'])

spidermap('MrBeast')

word_cloud('MrBeast')