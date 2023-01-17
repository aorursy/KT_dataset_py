import numpy as np

import pandas as pd



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns



import nltk

from PIL import Image

from wordcloud import WordCloud,STOPWORDS

from nltk.corpus import stopwords



import re



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/data-science-youtube-video-meta-data/data-science-youtube-channel-videos-metadata.csv')
data.head(2)
data.tail(2)
print('='*50)

print("Columns in data")

print('='*50)

print(data.columns.values)
print(f"Total number of samples in data are : {data.shape[0]}")

print('='*50)

print(f"Total number of features in data are : {data.shape[1]}")
data.info()
data.describe()
data.isna().sum()
data['viewCount'] = data['viewCount'].astype(float)
publish_time = pd.to_datetime(data['publishedAt'], format='%Y-%m-%dT%H:%M:%S%fZ')

data['publish_date'] = publish_time.dt.date

data['publish_time'] = publish_time.dt.time

data['publish_hour'] = publish_time.dt.hour
# Clean the data

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^(a-zA-Z)\s]','', text)

    return text

data['videoTitle'] = data['videoTitle'].apply(clean_text)
data['videoDescription'] = data['videoDescription'].apply(clean_text)
data.head(2)
publish_h = [0] * 24



for index, row in data.iterrows():

    publish_h[row["publish_hour"]] += 1

    

values = publish_h

ind = np.arange(len(values))





# Creating new plot

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

ax.yaxis.grid()

ax.xaxis.grid()

bars = ax.bar(ind, values)



# Sampling of Colormap

for i, b in enumerate(bars):

    b.set_color(plt.cm.viridis((values[i] - min(values))/(max(values)- min(values))))

    

plt.ylabel('Number of published videos', fontsize=20)

plt.xlabel('Time of publishing', fontsize=20)

plt.title('When most of the videos are published?', fontsize=35, fontweight='bold')

plt.xticks(np.arange(0, len(ind), len(ind)/6), [0, 4, 8, 12, 16, 20])



plt.show()
plt.figure(figsize = (12,6))



plt.subplot(221)

g1 = sns.distplot(data['viewCount'])

g1.set_title("VIEWS DISTRIBUITION", fontsize=16)



plt.subplot(224)

g2 = sns.distplot(data['likeCount'],color='green')

g2.set_title('LIKES DISTRIBUITION', fontsize=16)



plt.subplot(223)

g3 = sns.distplot(data['dislikeCount'], color='r')

g3.set_title("DISLIKES DISTRIBUITION", fontsize=16)



plt.subplot(222)

g4 = sns.distplot(data['commentCount'])

g4.set_title("COMMENTS DISTRIBUITION", fontsize=16)



plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)



plt.show()
data['viewCount_log'] = np.log(data['viewCount'] + 1)

data['likeCount_log'] = np.log(data['likeCount'] + 1)

data['dislikeCount_log'] = np.log(data['dislikeCount'] + 1)

data['commentCount_log'] = np.log(data['commentCount'] + 1)
plt.figure(figsize = (12,6))



plt.subplot(221)

g1 = sns.distplot(np.log(data['viewCount_log'] + 1))

g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)



plt.subplot(222)

g2 = sns.distplot(np.log(data['likeCount_log']+ 1),color='green')

g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)



plt.subplot(223)

g3 = sns.distplot(np.log(data['dislikeCount_log']+ 1), color='r')

g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)



plt.subplot(224)

g4 = sns.distplot(np.log(data['commentCount_log']+ 1))

g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)



plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)



plt.show()
plt.figure(figsize = (14,9))



plt.subplot(211)

g = sns.countplot('videoCategoryLabel', data=data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Counting the Video Category's ", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Count", fontsize=12)



plt.subplot(212)

g1 = sns.boxplot(x='videoCategoryLabel', y='viewCount_log', data=data, palette="Set1")

g1.set_xticklabels(g.get_xticklabels(),rotation=45)

g1.set_title("Views Distribuition by Category Names", fontsize=20)

g1.set_xlabel("", fontsize=15)

g1.set_ylabel("Views(log)", fontsize=15)



plt.subplots_adjust(hspace = 0.9, top = 0.9)



plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='videoCategoryLabel', y='likeCount_log', data=data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Likes Distribuition by Category Names ", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Likes(log)", fontsize=12)

plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='videoCategoryLabel', y='dislikeCount_log', data=data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Dislikes Distribuition by Category Names ", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Dislikes(log)", fontsize=12)

plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='videoCategoryLabel', y='commentCount_log', data=data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Comments Distribuition by Category Names", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Comments Count(log)", fontsize=12)



plt.show()
data.info()
data['like_rate'] =  data['likeCount'] / data['viewCount'] * 100

data['dislike_rate'] =  data['dislikeCount'] / data['viewCount'] * 100

data['comment_rate'] =  data['commentCount'] / data['viewCount'] * 100
data = data.replace([np.inf, -np.inf], np.nan)
plt.figure(figsize = (9,6))



g1 = sns.distplot(data['dislike_rate'], color='red',hist=False, label="Dislike")

g1 = sns.distplot(data['like_rate'], color='green',hist=False, label="Like")

g1 = sns.distplot(data['comment_rate'],hist=False,label="Comment")

g1.set_title('CONVERT RATE DISTRIBUITION', fontsize=16)

plt.xlabel('rate')

plt.legend()

plt.show()
plt.figure(figsize = (10,8))



#Let's verify the correlation of each value

sns.heatmap(data.corr(), annot=True)

plt.show()
publish_h = [0] * 24



for index, row in us_videos_first.iterrows():

    publish_h[row["publish_hour"]] += 1

    

values = publish_h

ind = np.arange(len(values))





# Creating new plot

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

ax.yaxis.grid()

ax.xaxis.grid()

bars = ax.bar(ind, values)



# Sampling of Colormap

for i, b in enumerate(bars):

    b.set_color(plt.cm.viridis((values[i] - min(values))/(max(values)- min(values))))

    

plt.ylabel('Number of videos that got trending', fontsize=20)

plt.xlabel('Time of publishing', fontsize=20)

plt.title('Best time to publish video', fontsize=35, fontweight='bold')

plt.xticks(np.arange(0, len(ind), len(ind)/6), [0, 4, 8, 12, 16, 20])



plt.show()
sns.distplot(data['like_rate'])
sns.countplot('videoCategoryId', data=data)

plt.show()
data['videoCategoryId'].value_counts()
def build_wordcloud(data, title):

    wordcloud = WordCloud(

        background_color='gray', 

        stopwords=set(STOPWORDS), 

        max_words=500, 

        max_font_size=40, 

        random_state=666

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    fig.suptitle(title, fontsize=16)

    fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
build_wordcloud(data['videoTitle'], 'Prevalent words in title')
build_wordcloud(data['videoDescription'], 'Prevalent words in video description')