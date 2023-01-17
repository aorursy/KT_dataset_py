# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import nltk
from nltk.corpus import stopwords
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  

import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
#df = pd.read_csv("songdata.csv")
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/songdata.csv')
df.head(10)
df.describe()
df = df.drop(['link'], axis=1)
fig=plt.figure(figsize=(18,6), dpi= 200)
df.groupby(['artist'])['text'].count().head(30).sort_values(ascending=False).plot(kind='bar')
## Removing white-space / new line
mytext = df[df['artist'] == 'Adele']['text'].replace(r'\s', ' ', regex=True)

#mytext = df.text.replace(r'\s', ' ', regex=True)

from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from stop_words import get_stop_words
import re
lyrics = mytext.str.cat(sep=' ').lower()
lyrics = re.sub('[^a-z]+', ' ', lyrics)
lyrics_tokenize = nltk.word_tokenize(lyrics)

stop_words = list(get_stop_words('en'))
nltk_words = list(stopwords.words('english'))
stop_words.extend(nltk_words)

#filtered_words = [word for word in word_list if word not in stopwords.words('english')]
filtered_lyrics = [word for word in lyrics_tokenize if word not in stop_words]
filtered_lyrics

stemmed_lyrics = []
stemmer = PorterStemmer()
for w in filtered_lyrics:
    stemmed_lyrics.append(stemmer.stem(w))

nltk.FreqDist(stemmed_lyrics).most_common(10)

len(nltk.sent_tokenize(df.text[0]))
#df.text[0].strip().splitlines()
#df.text[0].count('\n')

df['line_count'] = df.apply(lambda _: '', axis=1)

def lines(lyrics_text):
    lines = len("".join([s for s in lyrics_text.strip().splitlines(True) if s.strip()]).splitlines())
    return lines

for i in range(len(df.text)):
    df['line_count'][i] = lines(df.text[i])
df['word_count'] = df.apply(lambda _: '', axis=1)

for i in range(len(df.text)):
    df.word_count[i] = len(nltk.word_tokenize(df.text[i]))
#df['word_count'] = df.apply(lambda x: len(nltk.word_tokenize(df.text[i]) for i in range len(df.text)))
df.head(20)
fig=plt.figure(figsize=(20,8), dpi= 200)
df.groupby(['artist'])['line_count'].sum().head(100).sort_values(ascending=False).plot(kind='bar')
fig=plt.figure(figsize=(20,8), dpi= 200)
df.groupby(['artist'])['word_count'].sum().head(100).sort_values(ascending=False).plot(kind='bar')
#wc = WordCloud.generate_from_frequencies(df.song, 10)

#plot.imshow(wc)
#plot.show()
songs = df.groupby(['song'])['song'].count().sort_values(ascending=False).head(200)

#songs = list(df.song)
wc = WordCloud(width=800, height=400).generate_from_frequencies(songs)

fig=plt.figure(figsize=(20,8),facecolor= 'k',dpi= 400)
plt.imshow(wc)
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()

