import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import chardet

import re

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

from pylab import rcParams

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec

from sklearn.manifold import TSNE
with open('/kaggle/input/top50spotify2019/top50.csv', 'rb') as f:

    result = chardet.detect(f.read())

    



songs_df = pd.read_csv("../input/top50spotify2019/top50.csv", encoding=result['encoding'])

lyrics_df = pd.read_csv("../input/songs-lyrics/Lyrics.csv")
songs_df.shape
songs_df.head()
songs_df = songs_df.drop('Unnamed: 0', axis = 1 )
songs_df.info()
lyrics_df.info()
lyrics_df.shape
lyrics_df.head()
songs_df.describe()
songs_categorical_cols = ['Track.Name','Artist.Name','Genre']
songs_df.describe()
rcParams['figure.figsize'] = 10, 20

songs_df.drop(songs_categorical_cols,axis=1).hist();
rcParams['figure.figsize'] = 8, 5

sns.heatmap(songs_df.drop(songs_categorical_cols,axis=1).corr());
sns.pairplot(songs_df.drop(songs_categorical_cols,axis=1));
## Popularity

sns.boxplot( y = songs_df["Popularity"]);
fig, ax = plt.subplots(1,3)

fig.subplots_adjust(hspace=0.6, wspace=0.6)



sns.boxplot( y = songs_df["Beats.Per.Minute"], ax=ax[0])

sns.boxplot( y = songs_df["Energy"], ax=ax[1])

sns.boxplot( y = songs_df["Danceability"], ax=ax[2])



fig.show()
fig, ax = plt.subplots(1,3)

fig.subplots_adjust(hspace=0.6, wspace=0.6)



sns.boxplot( y = songs_df["Loudness..dB.."], ax=ax[0])

sns.boxplot( y = songs_df["Liveness"], ax=ax[1])

sns.boxplot( y = songs_df["Valence."], ax=ax[2])



fig.show()
fig, ax = plt.subplots(1,3)

fig.subplots_adjust(hspace=0.8, wspace=0.8)



sns.boxplot( y = songs_df["Length."], ax=ax[0])

sns.boxplot( y = songs_df["Acousticness.."], ax=ax[1])

sns.boxplot( y = songs_df["Speechiness."], ax=ax[2])



fig.show()
rcParams['figure.figsize'] = 10, 8

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black').generate(str(songs_df.Genre.values))



plt.imshow(wordcloud, interpolation = 'bilinear');
rcParams['figure.figsize'] = 10, 8

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black').generate(str(songs_df['Artist.Name'].values))



plt.imshow(wordcloud, interpolation = 'bilinear');
dataset = pd.merge(songs_df, lyrics_df, left_on='Track.Name', right_on='Track.Name')

dataset['Lyrics'] = dataset['Lyrics'].astype(str)
sns.jointplot(x='Danceability', y='Popularity', 

              data=dataset, kind='scatter');
sns.jointplot(x='Acousticness..', y='Popularity', 

              data=dataset, kind='scatter');
sns.jointplot(x='Loudness..dB..', y='Popularity', 

              data=dataset, kind='scatter');
sns.jointplot(x='Liveness', y='Popularity', 

              data=dataset, kind='scatter');
rcParams['figure.figsize'] = 15, 8

popular_genre = dataset[['Genre','Popularity']].groupby('Genre').sum().sort_values(ascending=False,by='Popularity')[:10]



sns.barplot(x=popular_genre.index, y=popular_genre['Popularity'], data=popular_genre, palette = "pastel");
dataset['Lyrics'] = dataset['Lyrics'].str.lower().replace(r'\n',' ')
stop_words_en = list(stopwords.words("english"))

stop_words_es = list(stopwords.words("spanish"))

punctuations = list(string.punctuation)

forbidden = ['(',')',"'",',','oh',"'s", 'yo',"'ll", 'el', "'re","'m","oh-oh","'d", "n't", "``", "ooh", "uah", "'em", "'ve", "eh", "pa", "brr", "yeah"] 

stop_words_all = set(stop_words_en + stop_words_es + punctuations + forbidden)
def cleanse_text(tokens):

    return [i for i in tokens if ((i not in list(stop_words_all)) and (re.search(r'\d+', i) == None)) ]
songs = []

for song in dataset.Lyrics.values:

    songs.append(cleanse_text(word_tokenize(song)))
word2vec = Word2Vec(songs, min_count=5)
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show();
tsne_plot(word2vec)