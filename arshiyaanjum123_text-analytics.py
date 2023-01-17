# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
df=pd.read_csv("../input/youtube-new/USvideos.csv")
df.head()
print(df.nunique())
df.info()
df.shape
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (12,8))
plt.subplot(221)
p1=sns.distplot(df['views'])
#most of the trending videos have 5 million views or less
print(df[df['views']<1.5e6]['views'].count()/df['views'].count())
#70% of videos are under 1.5 million views
plt.subplot(222)
p2=sns.distplot(df['likes'])
#around 72% of the videos have less than 50000 likes
print(df[df['likes']<50000]['likes'].count()/df['likes'].count())
plt.subplot(223)
#around 88% of the videos have less than 5000 dislikes
p3=sns.distplot(df['dislikes'])
print(df[df['dislikes']<5000]['dislikes'].count()/df['dislikes'].count())
plt.subplot(224)
#around 72% of the videos have less than 5000 comments
p4=sns.distplot(df['comment_count'])
print(df[df['comment_count']<5000]['comment_count'].count()/df['comment_count'].count())
#We can see that they are not normally distributed
plt.show()
#To make them normally distributed
import numpy as np
df['views_log']=np.log(df['views']+1)
df['likes_log']=np.log(df['views']+1)
df['dislikes_log']=np.log(df['dislikes']+1)
df['comment_count_log']=np.log(df['comment_count']+1)
plt.figure(figsize=(12,6))
plt.subplot(221)
q1=sns.distplot(df['views_log'],color='blue')
plt.subplot(222)
q2=sns.distplot(df['likes_log'],color='green')
plt.subplot(223)
q3=sns.distplot(df['dislikes_log'],color='r')
plt.subplot(224)
q4=sns.distplot(df['comment_count_log'])
plt.show()
#Yayyy everything is normally distrubuted
#Now lets see the quantiles
print("Views Quantiles")
print(df['views'].quantile([.01,0.25,0.5,0.75,0.99]))
print(df['views_log'].quantile([.01,0.25,0.5,0.75,0.99]))
print(df['likes'].quantile([.01,0.25,0.5,0.75,0.99]))
print(df['likes_log'].quantile([.01,0.25,0.5,0.75,0.99]))
print(df['dislikes'].quantile([.01,0.25,0.5,0.75,0.99]))
print(df['dislikes_log'].quantile([.01,0.25,0.5,0.75,0.99]))
print(df['comment_count'].quantile([.01,0.25,0.5,0.75,0.99]))
print(df['comment_count_log'].quantile([.01,0.25,0.5,0.75,0.99]))
#everything is changed to one scale 
#creating a new category with category name which implies with category id
df['categoryname']=np.nan
df.loc[(df["category_id"] == 1),"category_name"] = 'Film and Animation'
df.loc[(df["category_id"] == 2),"category_name"] = 'Cars and Vehicles'
df.loc[(df["category_id"] == 10),"category_name"] = 'Music'
df.loc[(df["category_id"] == 15),"category_name"] = 'Pets and Animals'
df.loc[(df["category_id"] == 17),"category_name"] = 'Sport'
df.loc[(df["category_id"] == 19),"category_name"] = 'Travel and Events'
df.loc[(df["category_id"] == 20),"category_name"] = 'Gaming'
df.loc[(df["category_id"] == 22),"category_name"] = 'People and Blogs'
df.loc[(df["category_id"] == 23),"category_name"] = 'Comedy'
df.loc[(df["category_id"] == 24),"category_name"] = 'Entertainment'
df.loc[(df["category_id"] == 25),"category_name"] = 'News and Politics'
df.loc[(df["category_id"] == 26),"category_name"] = 'How to and Style'
df.loc[(df["category_id"] == 27),"category_name"] = 'Education'
df.loc[(df["category_id"] == 28),"category_name"] = 'Science and Technology'
df.loc[(df["category_id"] == 29),"category_name"] = 'Non Profits and Activism'
df.loc[(df["category_id"] == 25),"category_name"] = 'News & Politics'
df.head()
#box plots for transformed variables
print(df.category_name.value_counts())
plt.figure(figsize=(35,15))
plt.subplot(511)
g1=sns.countplot(df['category_name'],palette="Set1")
plt.subplot(512)
g2= sns.boxplot(x='category_name', y='views_log', data=df, palette="Set1")
plt.subplot(513)
g3=sns.boxplot(x='category_name',y='likes_log',data=df,palette="Set1")
plt.subplot(514)
g4=sns.boxplot(x='category_name',y='dislikes_log',data=df,palette="Set1")
plt.subplot(515)
g5=sns.boxplot(x='category_name',y='comment_count_log',data=df,palette="Set1")
plt.show()
#without transforming boxplots
plt.figure(figsize=(35,15))
plt.subplot(511)
g1=sns.countplot(df['category_name'],palette="Set1")
plt.subplot(512)
g2= sns.boxplot(x='category_name', y='views', data=df, palette="Set1")
plt.subplot(513)
g3=sns.boxplot(x='category_name',y='likes',data=df,palette="Set1")
plt.subplot(514)
g4=sns.boxplot(x='category_name',y='dislikes',data=df,palette="Set1")
plt.subplot(515)
g5=sns.boxplot(x='category_name',y='comment_count',data=df,palette="Set1")
plt.show()
df['like_rate'] =  df['likes'] / df['views'] * 100
df['dislike_rate'] = df['dislikes'] / df['views'] * 100
df['comment_rate'] = df['comment_count'] / df['views'] * 100
plt.figure(figsize = (9,6))
g1 = sns.distplot(df['dislike_rate'], color='red',hist=False, label="Dislike")
g2 = sns.distplot(df['like_rate'], color='green',hist=False, label="Like")
g3= sns.distplot(df['comment_rate'],hist=False,label="Comment")
g1.set_title('CONVERT RATE DISTRIBUITION', fontsize=16)
plt.show()
plt.figure(figsize = (14,8))
plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.subplot(2,2,1)
g = sns.countplot(x='comments_disabled', data=df)
g.set_title("Comments Disabled", fontsize=16)

plt.subplot(2,2,2)
g1 = sns.countplot(x='ratings_disabled', data=df)
g1.set_title("Rating Disabled", fontsize=16)

plt.subplot(2,2,3)
g2 = sns.countplot(x='video_error_or_removed', data=df)
g2.set_title("Video Error or Removed", fontsize=16)
plt.show()
plt.figure(figsize = (10,8))
#Let's verify the correlation of each value
sns.heatmap(df[['like_rate', 'dislike_rate', 'comment_rate','comment_count_log','views_log','likes_log','dislikes_log', "category_name"]].corr(),annot=True)
plt.show()
df.corr()
cdf= df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts() \
            .to_frame().reset_index() \
            .rename(columns={"index": "year", "trending_date": "No_of_videos"})
print(cdf)
# This extracts the value counts and resets the indexes
plt.figure(figsize=(12,6))
plt.subplot(211)
k=sns.barplot(x=cdf["year"],y=cdf["No_of_videos"])
plt.show()
#more than 2/3rd videos are released in 2018
df["trending_date"].apply(lambda x:'20'+x[:2]).value_counts(normalize=True)
#to find the description of non integer columns
df.describe(include = ['O'])
#There are 205 unique dates when the video is released
#There are 6351 videos which appeared in trending multiple times which are 2207 different channels
grouped=df.groupby(['video_id'])
groups = []
wanted_groups = []
for key, item in grouped:
    groups.append(grouped.get_group(key))

for g in groups:
    if len(g['title'].unique()) != 1:
        wanted_groups.append(g)

wanted_groups[0]

#Counting number of words in each title and tag
df['count_word']=df['title'].apply(lambda x: len(str(x).split()))
df['count_word_tag']=df['tags'].apply(lambda x: len(str(x).split()))
#Counting number of unique words in each title and tag
df['count_word_unique']=df['title'].apply(lambda x: len(set(str(x).split())))
df['count_word_tag_unique']=df['tags'].apply(lambda x: len(set(str(x).split())))
##Counting number of letters in each title and tag
df['count_letters']=df['title'].apply(lambda x:len(str(x)))
df['count_letters_tags']=df['tags'].apply(lambda x:len(str(x)))
#Counting number of punctuation
import string
df['count_punctuation']=df['title'].apply(lambda x:len([c for c in str(x) if c in string.punctuation]))
df['count_punctuation_tags']=df['tags'].apply(lambda x:len([c for c in str(x) if c in string.punctuation]))
#Counting capital letters
df['count_capitals']=df['title'].apply(lambda x:len([c for c in str(x).split() if c.isupper()]))
df['count_capitals_tags']=df['tags'].apply(lambda x:len([c for c in str(x).split() if c.isupper()]))
#Counting stop words
import nltk
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
df['count_stopwords']=df['title'].apply(lambda x:len([c for c in str(x).lower().split() if c in eng_stopwords]))
df['count_stopwords_tag']=df['tags'].apply(lambda x:len([c for c in str(x).lower().split() if c in eng_stopwords]))
#Counting average length of words
df['mean_word_len']=df['title'].apply(lambda x:np.mean([len(w) for w in str(x).split()]))
df['mean_word_len_tag']=df['tags'].apply(lambda x:np.mean([len(w) for w in str(x).split()]))
#unique word percentage
df['unique_word_per']=(df['count_word_unique']/df['count_word'])*100
df['unique_word_per_tag']=(df['count_word_tag_unique']/df['count_word_tag'])*100
#Punct percent in each comment:
df_yout['punct_percent']=df_yout['count_punctuations']*100/df_yout['count_word']
df_yout['punct_percent_tags']=df_yout['count_punctuations_tags']*100/df_yout['count_word_tags']

plt.figure(figsize = (12,18))

plt.subplot(421)
g1 = sns.distplot(df['count_word'],hist=False, label='Text')
g1 = sns.distplot(df['count_word_tag'],hist=False, label='Tags')
g1.set_title("COUNT WORDS DISTRIBUITION", fontsize=16)
plt.subplot(422)
g2 = sns.distplot(df['count_word_unique'],hist=False, label='Text')
g2 = sns.distplot(df['count_word_tag_unique'],hist=False, label='Tags')
g2.set_title("COUNT UNIQUE DISTRIBUITION", fontsize=16)
plt.subplot(423)
g3 = sns.distplot(df['count_letters'], hist=False, label='Text')
g3 = sns.distplot(df['count_letters_tags'], hist=False, label='Tags')
g3.set_title("COUNT LETTERS DISTRIBUITION", fontsize=16)
plt.subplot(424)
g4 = sns.distplot(df["count_punctuation"], hist=False, label='Text')
g4 = sns.distplot(df["count_punctuation_tags"],hist=False, label='Tags')
g4.set_xlim([-2,50])
g4.set_title('COUNT PONCTUATIONS DISTRIBUITION', fontsize=16)
plt.subplot(425)
g5 = sns.distplot(df["count_capitals"] , hist=False, label='Text')
g5 = sns.distplot(df["count_capitals_tags"] ,  hist=False, label='Tags')
g5.set_title('COUNT WORDS UPPER DISTRIBUITION', fontsize=16)
plt.subplot(426)
g7 = sns.distplot(df["count_stopwords"], hist=False, label='Title')
g7 = sns.distplot(df["count_stopwords_tag"], hist=False, label='Tags')
g7.set_title('STOPWORDS DISTRIBUITION', fontsize=16)

plt.subplot(427)
g8 = sns.distplot(df["mean_word_len"], hist=False, label='Text')
g8 = sns.distplot(df["mean_word_len_tag"],hist=False, label='Tags')
g8.set_xlim([-2,100])
g8.set_title('MEAN WORD LEN DISTRIBUITION', fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.legend()
plt.show()
plt.figure(figsize = (12,8))
plt.subplot(221)
g=sns.boxplot(x='count_punctuation', y='views_log',data=df)
g.set_title("Views by Punctuations")
g.set_xlabel("Numer of Punctuations")
g.set_ylabel("Vews log")
plt.subplot(222)
g1 = sns.boxplot(x='count_punctuation', y='likes_log',data=df)
g1.set_title("Likes by Punctuations")
g1.set_xlabel("Numer of Punctuations")
g1.set_ylabel("Likes log")
plt.subplot(223)
g2 = sns.boxplot(x='count_punctuation', y='dislikes_log',data=df)
g2.set_title("Dislikes by Punctuations")
g2.set_xlabel("Numer of Punctuations")
g2.set_ylabel("Dislikes log")
plt.subplot(224)
g3 = sns.boxplot(x='count_punctuation', y='comment_count_log',data=df)
g3.set_title("Comments by Punctuations")
g3.set_xlabel("Numer of Punctuations")
g3.set_ylabel("Comments log")
plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.show()
plt.figure(figsize = (12,8))
plt.subplot(221)
g=sns.boxplot(x='count_punctuation_tags', y='views_log',data=df[df['count_punctuation_tags']<20])
g.set_title("Views by Punctuations")
g.set_xlabel("Numer of Punctuations")
g.set_ylabel("Vews log")
plt.subplot(222)
g1 = sns.boxplot(x='count_punctuation_tags', y='likes_log',data=df[df['count_punctuation_tags']<20])
g1.set_title("Likes by Punctuations")
g1.set_xlabel("Numer of Punctuations")
g1.set_ylabel("Likes log")
plt.subplot(223)
g2 = sns.boxplot(x='count_punctuation_tags', y='dislikes_log',data=df[df['count_punctuation_tags']<20])
g2.set_xlabel("Numer of Punctuations")
g2.set_ylabel("Dislikes log")
plt.subplot(224)
g3 = sns.boxplot(x='count_punctuation_tags', y='comment_count_log',data=df[df['count_punctuation_tags']<20])
g3.set_title("Comments by Punctuations")
g3.set_xlabel("Numer of Punctuations")
g3.set_ylabel("Comments log")
plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.show()
plt.figure(figsize=(25,20))
sns.heatmap(df.corr(),annot=True)
plt.show()
#printing wordcloud for title
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
stopwords=set(STOPWORDS)
wordcloud=WordCloud(background_color='Red',stopwords=stopwords,max_words=1000,max_font_size=120,random_state=42).generate(str(df['title']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TITLES")
plt.axis('off')
plt.show()
#printing wordcloud for tags
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
stopwords=set(STOPWORDS)
wordcloud=WordCloud(background_color='White',stopwords=stopwords,max_words=1000,max_font_size=120,random_state=42).generate(str(df['tags']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - Tags")
plt.axis('off')
plt.show()
#printing wordcloud for title
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
stopwords=set(STOPWORDS)
newStopWords= ['https', 'youtube', 'VIDEO','youtu','CHANNEL', 'WATCH']
stopwords.update(newStopWords)
wordcloud=WordCloud(background_color='Black',stopwords=stopwords,max_words=1000,max_font_size=120,random_state=42).generate(str(df['description']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - TITLES")
plt.axis('off')
plt.show()
df['publish_time']=pd.to_datetime(df['publish_time'],format='%Y-%m-%dT%H:%M:%S.%fZ')
df['month'] = df['publish_time'].dt.month

print("Category Name count")
print(df['month'].value_counts()[:5])

plt.figure(figsize = (14,9))

plt.subplot(211)
g = sns.countplot('month', data=df, palette="Set1")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Counting Months ", fontsize=20)
g.set_xlabel("Months", fontsize=15)
g.set_ylabel("Count", fontsize=15)

plt.subplot(212)
g1 = sns.boxplot(x='month', y='like_rate', data=df, palette="Set1")
g1.set_xticklabels(g.get_xticklabels(),rotation=45)
g1.set_title("Like Rate by Month", fontsize=20)
g1.set_xlabel("Months", fontsize=15)
g1.set_ylabel("Like Rate(log)", fontsize=15)

plt.subplots_adjust(hspace = 0.5, top = 0.9)

plt.show()
import datetime
#Publishing day
df["publishing_day"] = df["publish_time"].apply(lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
#publishing hour
df["publishing_hour"] = df["publish_time"].apply(lambda x: x[11:13])
df.drop(labels='publish_time', axis=1, inplace=True)
      
cdf = df["publishing_day"].value_counts()\
        .to_frame().reset_index().rename(columns={"index": "publishing_day", "publishing_day": "No_of_videos"})
g1= sns.barplot(x="publishing_day", y="No_of_videos", data=cdf)
plt.show()

cdf = df["publishing_hour"].value_counts().to_frame().reset_index()\
        .rename(columns={"index": "publishing_hour", "publishing_hour": "No_of_videos"})
k1=sns.barplot(x="publishing_hour",y="No_of_videos",data=cdf)
plt.show()
sizes=df["video_error_or_removed"].value_counts()
patches, texts = plt.pie(sizes,autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()
plt.show()