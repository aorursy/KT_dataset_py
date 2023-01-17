import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.rc('axes.formatter', useoffset=False)
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/tumbzilla_labels.csv')
df.head(2)
df.info()
df = df.drop(['Unnamed: 0'],axis=1)
df.quality.value_counts()
df['title'].loc[df['title'].str.contains('سكس نار')]
for col in ['title','categories','tags']:
    df[col] = df[col].str.replace('سكس نار','Firefighters') # thanks to google translator
    df[col] = df[col].str.replace('\d+','') # unnecessary numbers
    df[col] = df[col].str.replace(r'[_,-]+',' ') # too many underscores and hyphens in the data
    df[col] = df[col].str.replace(r'\bScene\b','').str.strip() # Scene word appears too many times. Redundant info
    
df.head(2)
most_viewed_title = df[['nb_views','title']].sort_values(by='nb_views', ascending=False).head(10)
most_viewed_title.sort_values(by='nb_views',ascending=True, inplace=True)

plt.figure(figsize=(15,10))
ax=sns.barplot(
    y=most_viewed_title['title'], 
    x=most_viewed_title['nb_views']
)
plt.xticks(rotation= 90)
plt.ylabel('Video Titles')
plt.xlabel('Number of views')
plt.title('Most viewed videos')
most_viewed_cat = df[['nb_views','categories']].sort_values(by='nb_views', ascending=False).head(10)
most_viewed_cat.sort_values(by='nb_views',ascending=True, inplace=True)

plt.figure(figsize=(15,10))
ax=sns.barplot(
    y=most_viewed_cat['categories'], 
    x=most_viewed_cat['nb_views']
)
plt.xticks(rotation= 90)
plt.ylabel('Categories')
plt.xlabel('Number of views')
plt.title('Most viewed categories')
most_voted_cat = df[['voting','title']].sort_values(by='voting', ascending=False).head(20)
most_voted_cat.sort_values(by='voting',ascending=True, inplace=True)

plt.figure(figsize=(15,10))
ax=sns.barplot(
    y=most_voted_cat['title'], 
    x=most_voted_cat['voting']
)
plt.xticks(rotation= 90)
plt.ylabel('Video Titles')
plt.xlabel('Votings')
plt.title('Most voted videos')
print(df['voting'].mean())
print(df['voting'].max())
plt.hist(df['voting'])
plt.show()
df2 = df[['nb_views','length']].sort_values(by='nb_views', ascending=False).head(100)

plt.figure(figsize=(15,10))
ax=sns.barplot(
    y=df2['nb_views'], 
    x=df2['length'],
    ci=None
)
plt.xticks(rotation= 90)
plt.ylabel('Number of views')
plt.xlabel('Length of videos')
plt.title('Length of videos x Number of views')
plt.figure(figsize=(12,8))
labels = ['HD viewers','Low Quality viewers']
sizes = [
    np.array(df['nb_views'].loc[df['quality']=='HD']).sum(),
    np.array(df['nb_views'].loc[df['quality']=='LOW']).sum()
]
colors = ['gold', 'lightgreen']
explode = (0.05, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df2 = df[['voting','length']].sort_values(by='voting', ascending=False).head(100)

plt.figure(figsize=(15,10))
ax=sns.barplot(
    y=df2['voting'], 
    x=df2['length'],
    ci=None
)
plt.xticks(rotation= 90)
plt.ylabel('Votings')
plt.xlabel('Length of videos')
plt.title('Length of videos x Votings')
plt.figure(figsize=(12,8))
labels = ['HD video voters','Low Quality video voters']
sizes = [
    np.array(df['voting'].loc[df['quality']=='HD']).sum(),
    np.array(df['voting'].loc[df['quality']=='LOW']).sum()
]
colors = ['blue', 'lightcoral']
explode = (0.05, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=40)
 
plt.axis('equal')
plt.show()
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=600,
                          random_state=69,
                         ).generate(" ".join(df['tags'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("Most used words in Tags - Word Cloud", fontsize=18)
plt.axis('off')
plt.show()
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=600,
                          random_state=69,
                         ).generate(" ".join(df['categories'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("Most used words in Categories - Word Cloud", fontsize=18)
plt.axis('off')
plt.show()
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=600,
                          random_state=69,
                         ).generate(" ".join(df['title'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("Most used words in Titles - Word Cloud", fontsize=18)
plt.axis('off')
plt.show()
