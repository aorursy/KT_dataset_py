# BASIC NLP ANALYSIS TO A SINGLE SONG FROM THE DATASET
# @nathalyAlarconT

import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
% matplotlib inline
filename = '../input/billboard_lyrics_1964-2015.csv'
df = pd.read_csv(filename, encoding='latin-1' )
df.head()
df.dtypes
df.shape
print(df['Artist'].nunique())
df['Artist'].value_counts()
df['Artist'].value_counts()[:30].plot('bar')
artist_name = 'the killers'
artistSongs = df[df['Artist'] == artist_name]
artistSongs
my_song = str(artistSongs['Lyrics'].values[0])
my_song
import nltk
from nltk import FreqDist
data = nltk.word_tokenize(my_song)
fdist=FreqDist(data)
print(fdist)
print(fdist.most_common(100))
# pd.DataFrame(fdist.most_common(100)).plot('bar')
fdist.plot(50, cumulative=True)
fdist.plot(30, cumulative=False)
from nltk.tokenize import sent_tokenize, word_tokenize
print(sent_tokenize(my_song))  # I don't have good results here because the lyrics 

print("----------------------------------------------")
print(word_tokenize(my_song))
# Removing Stop Words 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words("english"))
# print(stop_words)

words = word_tokenize(my_song)
filter_sentence = []
filter_sentence.append([w for w in words if w not in stop_words])
print(filter_sentence)
# Lematization 

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()

# exa = word_tokenize(filter_sentence)
stem_sentence = []
for w in filter_sentence[0]:
    stemText = ps.stem(w)
    stem_sentence.append(stemText)
mostCommonPhrases = pd.DataFrame(stem_sentence, columns=['words'])
mostCommonPhrases['words'].value_counts()[:40].plot('bar')

