import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

import nltk as nlp

import warnings

warnings.filterwarnings("ignore")
#usa



df=pd.read_csv("../input/youtube-new/USvideos.csv")
df.head()
df.drop(["video_id","thumbnail_link","trending_date","publish_time","category_id","description","channel_title"],axis=1,inplace=True)
df.head()
df.tail()
df.info()
df.iloc[:,3]
#comments_disabled	ratings_disabled	video_error_or_removed	

from sklearn.preprocessing import LabelEncoder

labelEncoder_Y=LabelEncoder()

df.iloc[:,6]=labelEncoder_Y.fit_transform(df.iloc[:,6].values)

df.iloc[:,7]=labelEncoder_Y.fit_transform(df.iloc[:,7].values)

df.iloc[:,8]=labelEncoder_Y.fit_transform(df.iloc[:,8].values)
df.info()
df.head()
title_list=[]

for title in df.title:

    title=re.sub("[^a-zA-Z]"," ",title)

    title=title.lower()

    title=nltk.word_tokenize(title)

    lemma=nlp.WordNetLemmatizer()

    title=[lemma.lemmatize(word) for word in title]

    title=" ".join(title)

    title_list.append(title)
#bag of words 

from sklearn.feature_extraction.text import CountVectorizer

max_features=150

count_vectorizer=CountVectorizer(max_features=max_features, stop_words="english")

sparce_matrix=count_vectorizer.fit_transform(title_list).toarray()

print("The 150 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
tags_list=[]

for tags in df.tags:

    tags=re.sub("[^a-zA-Z]"," ",tags)

    tags=tags.lower()

    tags=nltk.word_tokenize(tags)

    lemma=nlp.WordNetLemmatizer()

    tags=[lemma.lemmatize(word) for word in tags]

    tags=" ".join(tags)

    tags_list.append(tags)
max_features=150

count_vectorizer=CountVectorizer(max_features=max_features, stop_words="english")

sparce_matrix=count_vectorizer.fit_transform(tags_list).toarray()

print("The 150 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
df.head()
df = df.sort_values(by=["likes"], ascending=False)

df['rank']=tuple(zip(df.likes))

df['rank']=df.groupby('likes',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values

df.drop(["rank"],axis=1,inplace=True)

df.reset_index(inplace=True,drop=True)

df.head()
df.tail()
plt.figure(figsize=(18,5))

ax=df.likes.plot.kde(bw_method=0.7)

ax=df.dislikes.plot.kde(bw_method=0.5)

ax=df.comment_count.plot.kde(bw_method=0.7)

ax.legend()

plt.show()