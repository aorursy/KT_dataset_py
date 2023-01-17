# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/songlyrics/songdata.csv')
df.head()
df.describe()
df['text'] = df['text'].str.replace('\n', '')
df.head()
res = df[["artist", "song"]].groupby(['artist']).count().sort_values('song',ascending = False)
res.describe()
res['song']
#word


# distribution of number of songs and no of artists 

plt.figure(figsize=(10, 4))

sns.distplot(res, bins = 10, kde = False)

plt.xlabel('No. of songs', fontsize=16)

plt.ylabel('No. or artists', fontsize=16)
import matplotlib.pyplot as plt
#res.sort()
print(res)
#generating wordcloud of artists

from wordcloud import WordCloud

wc = WordCloud(background_color="white")

wc.generate_from_frequencies(res['song'])

plt.imshow(wc, interpolation="bilinear")

plt.axis("off")

plt.show()
#creating two new columns

df['tokenized_text_unchanged'] = df['text'].str.split()

df['tokenized_text'] = df['text'].str.split()
df.head(1)
#removing stop words from tokenized words

from nltk.corpus import stopwords

stop = stopwords.words('english')

removestopwords = ['only','between','yourself','must','myself','again','once','both']

for word in list(stop):  # iterating on a copy since removing will mess things up

    if word in removestopwords:

        stop.remove(word)
#stop
# lowering the sting

df['tokenized_text'] = df['text'].str.lower().str.split()
df['tokenized_text']
# remove the words that are part of the stop words

df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [item for item in x if item not in stop])
df
df['tokenized_text'][100]
# remove words being used multiple times in data set

for x in range(len(df['tokenized_text'])):

    df['tokenized_text'][x] = list(set(df['tokenized_text'][x]))
df['tokenized_text']
# remove the characters with less than 3 characters

df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [item for item in x if len(item) > 3])
#df['tokenized_text']
# removing the special characters

import re

for x in range(len(df['tokenized_text'])):

    res_series = pd.Series(df['tokenized_text'][x]).str.replace(r'[\W]', '')

    df['tokenized_text'][x] = res_series.values.tolist()
#df['tokenized_text'][3]
#df
#df.groupby(['song']).agg(

#word_count = pd.NamedAgg(column='text' , aggfunc = 'count' )

#)
#df.assign(count = df['tokenized_text'].str.len()).groupby('song', as_index=False)['count']
#count = df['tokenized_text'].str.len()

#count
df = df.assign(count = df['tokenized_text'].str.len())
#finding out the songs with maximum and mimimum words

songs_with_word_result = df[['song','count']].sort_values('count', ascending = False).head(10)

songs_with_word_result_less = df[['song','count']].sort_values('count', ascending = False).tail(10)
#plotting songs of maximum words

plt.figure(figsize=(8, 3))

chart = sns.barplot(x='song',y = 'count', data=songs_with_word_result, palette="rocket")

for item in chart.get_xticklabels():

    item.set_rotation(45)
#plotting songs of mimimum words

plt.figure(figsize=(8, 3))

chart = sns.barplot(x='song',y = 'count', data=songs_with_word_result_less, palette="rocket")

for item in chart.get_xticklabels():

    item.set_rotation(45)
#.sum().sort_values(['count'],ascending = False).head(10)
#df.groupby('song').count().sort_values('count', ascending = False).head(20)
# count the number of words in a song, can include sings that are repeated multiple times in a database

df.assign(count = df['tokenized_text'].str.len()).groupby('song', as_index=False)['count'].sum().sort_values(['count'],ascending = False).head(10)
df
# count of words for unchanged lyrics

df = df.assign(unchanged_count = df['tokenized_text_unchanged'].str.len())
df
# sorting values on the basis of number of words for a song , without cleaning the lyrics

df[['song','unchanged_count']].sort_values('unchanged_count', ascending = False).head(10)
# number of times a word is repeated in the lyrics without cleaning them 

#df.text.str.split(expand=True).stack().value_counts()
df['tokenized_joined_text'] = df['tokenized_text'].str.join(" ")
#df.head(2)
#df['tokenized_text'][0]
# number of times a word is repeated after cleaning the lyrics

df.tokenized_joined_text.str.split(expand=True).stack().value_counts()
#loading the lexicon file

df_lexicon = pd.read_csv("../input/nrc-lexicon/NRC-Emotion-Lexicon.csv")
df_lexicon.head(2)
df_lexicon = df_lexicon.rename(columns={"English (en)": "word"})
#df_lexicon.head(1)
#df_lexicon[df_lexicon['word'] == 'charitable']
#df_lexicon[(df_lexicon.Anger == 1)]
words_used = df.tokenized_joined_text.str.split(expand=True).stack()
#words_used
#df_inner = pd.merge(df1, words_used, on='words_us', how='inner')
#df_lexicon[df_lexicon.word == words_used]
#len(list(df.tokenized_joined_text.str.split(expand=True).stack().unique()))
#len(list(df.tokenized_joined_text.str.split(expand=True).stack().value_counts()))
#list( df.tokenized_joined_text.str.split(expand=True).stack().unique() ,df.tokenized_joined_text.str.split(expand=True).stack().value_counts())
df.isnull().values.any()
lst = list(df.tokenized_joined_text.str.split(expand=True).stack().value_counts())

lst2 = list(df.tokenized_joined_text.str.split(expand=True).stack().unique())
df_words = pd.DataFrame(list(zip(lst2, lst)), columns= ['word' , 'count'])
df_words
df_lexicon[df_lexicon['Anger'] == 1]['word']
#df_words.loc[df_words['word'] == list(df_lexicon['word'])]
df_lexicon.describe()
anger = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Anger'] == 1]['word'])) == True ].head(10)

anticipation = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Anticipation'] == 1]['word'])) == True ].head(10)

disgust = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Disgust'] == 1]['word'])) == True ].head(10)

fear = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Fear'] == 1]['word'])) == True ].head(10)

joy = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Joy'] == 1]['word'])) == True ].head(10)

sadness = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Sadness'] == 1]['word'])) == True ].head(10)

surprise = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Surprise'] == 1]['word'])) == True ].head(10)

trust = df_words[df_words['word'].isin(list(df_lexicon[df_lexicon['Trust'] == 1]['word'])) == True ].head(10)
anger.head(2)
f, axes = plt.subplots(4, 2, figsize=(12, 10))

chart = (sns.barplot(x='count',y='word', data=anger, ax=axes[0, 0],palette="rocket"),

sns.barplot(x='count',y='word', data=anticipation, ax=axes[0, 1],palette="rocket"),

sns.barplot(x='count',y='word', data=disgust, ax=axes[1, 0],palette="rocket"),

sns.barplot(x='count',y='word', data=fear, ax=axes[1, 1],palette="rocket"),

sns.barplot(x='count',y='word', data=joy, ax=axes[2, 0],palette="rocket"),

sns.barplot(x='count',y='word', data=sadness, ax=axes[2, 1],palette="rocket"),

sns.barplot(x='count',y='word', data=surprise, ax=axes[3, 0],palette="rocket"),

sns.barplot(x='count',y='word', data=trust, ax=axes[3, 1],palette="rocket"))

df_lexicon[df_lexicon['word'] == 'hoot']