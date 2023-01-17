!pip install lyrics_extractor
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

from lyrics_extractor import Song_Lyrics

import nltk

from nltk.corpus import stopwords

from textblob import TextBlob

from collections import Counter
import cufflinks as cf

import chart_studio.plotly



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go



init_notebook_mode(connected=True)

cf.go_offline()
sns.set_style('darkgrid')

sns.set_color_codes("pastel")
spotify = pd.read_csv("../input/spotify-top-50-w-lyrics/spotify_w_lyrics.csv",  encoding = "ISO-8859-1" , index_col= 0)
spotify.head(10)
spotify.columns = [cols.replace('.', '') for cols in spotify.columns]

spotify = spotify.sort_values(by = 'Popularity', ascending =False).reset_index(drop = True)
spotify.describe()
spotify.info()
# target variable: popularity 

sns.distplot(spotify['Popularity'], kde = False, bins = 10)
# distribution of other features



x_cols = ['BeatsPerMinute', 'Energy','Danceability', 'LoudnessdB', 'Liveness', 'Valence', 'Length','Acousticness', 

          'Speechiness']



fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15,15))

for i, x_col in enumerate(x_cols):

    sns.distplot( spotify[x_col], ax=axes[i//3,i%3], kde = False, bins = 10)

    axes[i//3,i%3].set_xlabel(x_col) 
# How does features relate to popularity



x_cols = ['BeatsPerMinute', 'Energy','Danceability', 'LoudnessdB', 'Liveness', 'Valence', 'Length','Acousticness', 

          'Speechiness']



fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15,15))

for i, x_col in enumerate(x_cols):

    sns.regplot( x = x_col,  y = 'Popularity', ax=axes[i//3,i%3], data = spotify)

    axes[i//3,i%3].set_xlabel(x_col) 
# correlation heatmap between X features and popularity



fig = plt.figure(figsize = (10,8))



mask = np.zeros_like(spotify.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(spotify.corr(), annot = True,linewidths = 0.3, mask = mask)
# Artist Popularity:

fig = plt.figure(figsize = (15,8))

sns.set_style('whitegrid')

artist = spotify.groupby('ArtistName').size().reset_index(name = 'count')

artist = artist.sort_values(by = 'count', ascending =False)

sns.barplot(y = 'ArtistName',x="count", data=artist,  color="b")

sns.despine(left=True, bottom=True)
# Genre Popularity:



spotify['Genre'].unique()
def parent_genre(genre):

    music_genre = {'electronic' : ['electropop','trap music','pop house', 'big room', 'brostep' ,'edm'],

                   'hip hop/rap': ['canadian hip hop','atl hip hop','reggaeton','reggaeton flow','dfw rap',

                                   'country rap'],

                   'pop': ['pop','panamanian pop', 'canadian pop', 'australian pop', 'dance pop', 'boy band'],

                   'others': ['escape room', 'latin','r&b en espanol']}

    

    for parent, sub in music_genre.items():

        if genre in sub:

            return parent
spotify['parent_genre'] = spotify['Genre'].apply(parent_genre)
# Genre Popularity:

import plotly.graph_objects as go



artist = spotify.groupby('parent_genre').size().reset_index(name = 'count')

artist = artist.sort_values(by = 'count', ascending =False)



values = artist['count'].tolist()

labels = artist['parent_genre'].tolist()



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(textposition='inside', textinfo='value+label', title_text = 'Parent Genre Segmentation')

fig.show()
parent_genre = pd.get_dummies(spotify['parent_genre'],drop_first=True)
spotify = pd.concat([spotify, parent_genre],axis=1)
spotify.head()
# Original function that scrap for song lyrics

# def get_lyrics(track):

#     extract_lyrics = Song_Lyrics('GCS_API_KEY', 'GCS_ENGINE_ID')

#     song_title, song_lyrics = extract_lyrics.get_lyrics(track)

#     return song_lyrics
# spotify['lyrics'] = spotify['TrackName'].apply(lambda row: (get_lyrics(row)))

# spotify = spotify.replace('', 'None')
def pre_processing(lyrics):

    lyrics = lyrics.replace('\n', ' ',).lower()

    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    lyrics = re.sub(r'\(.*?\)', '', lyrics)

    lyrics = re.sub(r'\{.*?\}', '', lyrics)

    

    lyrics = re.sub(r'[^a-zA-Z0-9 ]', '', lyrics)

    lyrics = ' '.join(lyrics.split())

    return lyrics



def count_unique(df):

    text = df['Lyrics']

    stop_words = stopwords.words('english')

    newStopWords = ['youre','im', 'ill','ive', 'm', 'oh' , 'yeh', 'yeah', 'dont', 'got', 'gonna', 'wanna']

    stop_words.extend(newStopWords)

    stopwords_dict = Counter(stop_words)

    

    initial_len = len(text.split())

    clean_lyrics = ' '.join([word for word in text.split() if word not in stopwords_dict])

    text = set([word for word in text.split() if word not in stopwords_dict])

    unique_length = (len(text)/initial_len)*100

    return clean_lyrics, unique_length
spotify['Lyrics'] = spotify['lyrics'].apply(lambda lyrics:pre_processing(lyrics))

spotify['text_length'] = spotify['Lyrics'].apply(lambda lyrics: len(lyrics.split()))

spotify[['clean_lyrics','unique_length']] = spotify.apply(count_unique, result_type='expand', axis = 1)
spotify.head()
def sentiment_analysis(lyrics):

#   TextBlob has a function that allows for translation of text to eng, 

#   its still possible to run sentiment analysis even without translating the lyrics.

    blob = TextBlob(lyrics)

    language = blob.detect_language()

    if language != 'en':

        blob = blob.translate(to="en")



    for sentence in blob.sentences:

        sentiment = sentence.sentiment.polarity

    return sentiment
spotify['sentiment'] = spotify['clean_lyrics'].apply(sentiment_analysis)
lexicalrichness = spotify[spotify['text_length'] > 1]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (18,5))

sns.set_style('darkgrid')

cols = ['text_length', 'unique_length', 'sentiment']

 

for i, x_col in enumerate(cols):

    sns.distplot(lexicalrichness[x_col], ax=axes[i], kde = False, bins = 15)

    axes[i].set_xlabel(x_col) 
fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (18,5))

sns.set_style('darkgrid')

cols = ['text_length', 'unique_length', 'sentiment']

 

for i, x_col in enumerate(cols):

    sns.scatterplot(x = x_col, y = 'Popularity' , data = lexicalrichness, ax=axes[i])

    axes[i].set_xlabel(x_col) 
# wordcloud for most popular words 

# Here's a word cloud for those curious on popular words.

from wordcloud import WordCloud

def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))



    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(spotify['clean_lyrics'])
from sklearn.linear_model import LinearRegression

from sklearn import metrics

import statsmodels.api as sm

import matplotlib.lines as mlines

import matplotlib.transforms as mtransforms
spotify.columns
# we shall skip observation withtext length = 1 as it may skew our results. 

df = spotify[spotify['text_length'] > 1]
"""

To avoid the problems associated with co-linearity, only BeatsPerMinute is used 

as the dependent variable between the two. 

"""



y = df['Popularity']

X = df[['BeatsPerMinute','Valence','unique_length', 'hip hop/rap', 'others', 'pop']]
lm = LinearRegression()

lm.fit(X,y)

predictions = lm.predict(X)
X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))

sns.scatterplot(x = y,y = predictions, ax = ax1)

line = mlines.Line2D([0, 1], [0, 1], color='red')

transform = ax1.transAxes

line.set_transform(transform)

ax1.add_line(line)

ax1.set_xlim([70,100])

ax1.set_ylim([70,100])



sns.distplot((y-predictions),bins=10, ax= ax2);
MAE = metrics.mean_absolute_error(y, predictions)

MSE = metrics.mean_squared_error(y, predictions)

RMSE = np.sqrt(metrics.mean_squared_error(y, predictions))



error_df = pd.DataFrame(data = [MAE, MSE, RMSE], index = ['MAE', 'MSE', "RMSE"], columns=['Error'])

error_df