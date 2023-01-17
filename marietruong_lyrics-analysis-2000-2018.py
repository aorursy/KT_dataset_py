# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
lyrics = pd.read_csv('/kaggle/input/spotify/billboard_2000_2018_spotify_lyrics.csv', encoding = 'ISO-8859-1 ')
pd.options.display.max_columns = 100
lyrics
# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Natural Language Processing
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter
from textblob import TextBlob
from gensim import matutils, models, corpora
import scipy.sparse
from nltk import word_tokenize, pos_tag
lyrics['date'] = pd.to_datetime(lyrics.date)
lyrics.date = lyrics.date.astype(str)
lyrics.date = lyrics.date.str.split('-').str[0]
lyrics
x = lyrics.groupby('date').count().sort_values('title', ascending = False).index
y = lyrics.groupby('date').count().sort_values('title', ascending = False).title
plt.figure()
plt.title('Number of songs per year in the dataset')
plt.xticks(rotation = 'vertical')
sns.barplot(x, y)
plt.figure(figsize = (20,10))
plt.title('Artists that had the most number 1 songs')
plt.xticks(rotation = 'vertical')
x = lyrics[lyrics.peak_pos == 1 ].groupby('artist').count().sort_values('date', ascending = False).index[0:30]
y = lyrics[lyrics.peak_pos == 1 ].groupby('artist').count().sort_values('date', ascending = False).date[0:30]
sns.barplot(x, y)
plt.figure(figsize = (20,10))
plt.title('Artists that often reach TOP10')
plt.xticks(rotation = 'vertical')
x = lyrics[lyrics.peak_pos < 10 ].groupby('artist').count().sort_values('date', ascending = False).index[0:30]
y = lyrics[lyrics.peak_pos < 10 ].groupby('artist').count().sort_values('date', ascending = False).date[0:30]
sns.barplot(x, y)


lyrics.groupby(['date', 'artist']).count().sort_values(['date', 'title'], ascending = [True, False]).groupby('date').head(1).index
#y = lyrics.groupby('year').count().sort_values('date', ascending = False).date[0:30]
#sns.barplot(x, y)

plt.figure()
plt.title('Missing values')
sns.heatmap(lyrics.isna() == False)
lyrics = lyrics[['date', 'title', 'artist', 'duration_ms', 'lyrics']]
lyrics
lyrics = lyrics[lyrics.lyrics != 'Error: Could not find lyrics.']
lyrics = lyrics[lyrics.lyrics.isna() != True]
lyrics

length = lyrics[lyrics.duration_ms != 'unknown']
#lyrics['duration_ms'] = lyrics['duration_ms'].astype(float)
length.duration_ms = length.duration_ms.astype(float)
plt.figure(figsize = (20,10))
plt.title('Average song length of Billboard Hot 100')
plt.xticks(rotation = 'vertical')
x = length.groupby('date').mean().index
y = length.groupby('date').mean().duration_ms
sns.lineplot(x, y)
lyrics['length'] = 0
lyrics['length'] = lyrics.lyrics.str.len()
x = lyrics.groupby('date').mean().index
y = lyrics.groupby('date').mean().length
plt.figure(figsize = (15,9))
plt.title('Lyrics length')
plt.xticks(rotation = 'vertical')
sns.lineplot(x,y)

length['proportion'] = length.lyrics.str.len()
length.proportion = length.proportion/length.duration_ms
x = length.groupby('date').mean().index
y = length.groupby('date').mean().proportion
plt.figure(figsize = (15,9))
plt.title('Words importance')
plt.xticks(rotation = 'vertical')
sns.lineplot(x,y)
lyrics.lyrics = lyrics.lyrics.str.lower()
lyrics.lyrics = lyrics.lyrics.replace({'\n' : ' '}, regex = True)
lyrics.lyrics = lyrics.lyrics.map(lambda x : re.sub('\d','', str(x)))
lyrics
lyrics = lyrics.set_index('date')
x = list(lyrics.index.unique())
dct = {}
for i in x:
    vectorizer = CountVectorizer(stop_words = 'english')
    dtm = vectorizer.fit_transform(lyrics.loc[i].lyrics)
    dtm = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names())
    dct['year_%s' % i] = dtm

y = []
for i in x:
    unique_words = dct['year_%s' % i]
    unique_words[unique_words > 0] = 1
    y.append(unique_words.sum(axis = 1).mean())
plt.figure(figsize = (15,9))
plt.title('Amount of vocabulary (unique words)')
plt.xticks(rotation = 'vertical')
sns.lineplot(x,y)
y
y2 = list(lyrics.groupby('date').length.mean().sort_index(ascending = False).values)
Y = [x/y for x, y in zip(y, y2)]
plt.figure()
plt.title('Proportion of unique words')
plt.xticks(rotation = 'vertical')
sns.lineplot(x, Y)
total_lyrics =lyrics.groupby('date')['lyrics'].transform(lambda x: ','.join(x)).drop_duplicates()
total_lyrics =lyrics.groupby('date')['lyrics'].transform(lambda x: ','.join(x)).drop_duplicates()
total_lyrics

cv = CountVectorizer()
top_words = cv.fit_transform(total_lyrics)

top = pd.DataFrame(top_words.toarray(), columns = cv.get_feature_names())
#top = top.product(axis = 0)
#top[top != 0]
add_stop_words = set()
for i in top.index:
    add_stop_words =add_stop_words.union((top.loc[i].sort_values(ascending = False)[0:20].index))
#add_stop_words = list(top[top > 10000].index)
add_stop_words.discard('love')
add_stop_words.update(['don\'t', 'i\'m'])
add_stop_words
plt.figure(figsize = (20,20))
k = 1
for i in x :
    texte = total_lyrics.loc['%s' % i]
    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    wordcloud = WordCloud(stopwords = stop_words).generate(texte)
    plt.subplot(5,4,k)
    plt.title(i)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    k += 1
plt.show()

pol = lambda x: TextBlob(x).sentiment.polarity
lyrics['polarity'] = 0
lyrics['polarity'] = lyrics['lyrics'].apply(pol)
lyrics
x = lyrics.groupby('date').mean().index
y = lyrics.groupby('date').mean().polarity
plt.figure()
plt.title('Polarity of songs')
plt.xticks(rotation = 'vertical')
sns.lineplot(x,y )
top.index = total_lyrics.index
top
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)
lyrics['nouns_adj'] = 0 
lyrics['nouns_adj'] = lyrics.lyrics.apply(nouns_adj)
lyrics
lyrics_na =lyrics.groupby('date')['nouns_adj'].transform(lambda x: ','.join(x)).drop_duplicates()
lyrics_na
add_stop_words = {'baby', 'girl', 'man', 'hey', 'yeah', 'time', 'oh', 'life', 'little', 'good', 'right', 'day', 'cause','ya', 'ha'}
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
vect = TfidfVectorizer(stop_words = stop_words)
tfidf = vect.fit_transform(lyrics_na)
tfidf
nmf = NMF(n_components = 4)
topic_values = nmf.fit_transform(tfidf)
topic_values
for topic_num, topic in enumerate(nmf.components_):
    message = 'Topic {}:'.format(topic_num)
    message += ' '.join([vect.get_feature_names()[i] for i in topic.argsort()[:-20:-1]])
    print (message)
lyrics_2000 = lyrics.loc['2000']
lyrics_2000
stop_words = text.ENGLISH_STOP_WORDS
vect = TfidfVectorizer(stop_words = stop_words)
tfidf = vect.fit_transform(lyrics_2000.nouns_adj)
tfidf
nmf = NMF(n_components = 3)
topic_values = nmf.fit_transform(tfidf)
for topic_num, topic in enumerate(nmf.components_):
    message = 'Topic {}:'.format(topic_num)
    message += ' '.join([vect.get_feature_names()[i] for i in topic.argsort()[:-20:-1]])
    print (message)
nmf = NMF(n_components = 4)
topic_values = nmf.fit_transform(tfidf)
for topic_num, topic in enumerate(nmf.components_):
    message = 'Topic {}:'.format(topic_num)
    message += ' '.join([vect.get_feature_names()[i] for i in topic.argsort()[:-20:-1]])
    print (message)
nmf = NMF(n_components = 5)
topic_values = nmf.fit_transform(tfidf)
for topic_num, topic in enumerate(nmf.components_):
    message = 'Topic {}:'.format(topic_num)
    message += ' '.join([vect.get_feature_names()[i] for i in topic.argsort()[:-20:-1]])
    print (message)
x = lyrics_na.index
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
for i in x:
    tfidf = vect.transform(lyrics.loc[i].nouns_adj)
    y = nmf.transform(tfidf).mean(axis = 0)
    k = 0
    for t in [y1,y2,y3,y4,y5]:
        t.append(y[k])
        k += 1

plt.figure(figsize = (20,10))
plt.title('Topics evolution since 2000')
sns.lineplot(x,y1, label = 'Profanity, sex', color = 'red')
sns.lineplot(x,y2, label = 'Life and love', color = 'pink')
sns.lineplot(x,y3, label = 'Sadness', color = 'black')
sns.lineplot(x,y4, label = 'Flirting', color = 'green')
sns.lineplot(x,y5, label = 'Party', color = 'blue')
        
