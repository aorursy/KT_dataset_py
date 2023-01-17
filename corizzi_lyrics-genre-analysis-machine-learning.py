%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt

songList = pd.read_csv( "../input/lyrics.csv", index_col=0 ).sample(50000)
# Clean up the dataset
emptyLyrics = len(songList)
songList = songList[songList['lyrics']!='instrumental'].dropna()
emptyLyrics -= len(songList)
print(str(emptyLyrics) + " rows dropped (no lyrics).")

genreCount = songList['genre'].value_counts()
yearCount  = songList['year'].value_counts().head( 12 )

fig, axarr = plt.subplots(2, 2)
fig.tight_layout()


genreCount.plot.pie( figsize=(12, 12), fontsize=16, ax=axarr[0][0] , autopct='%.1f' )
genreCount.plot.bar( figsize=(22, 12), fontsize=16, ax=axarr[0][1] )

yearCount.plot.pie( figsize=(12, 12), fontsize=16, ax=axarr[1][0], autopct='%.1f' )
yearCount.plot.bar( figsize=(22, 12), fontsize=16, ax=axarr[1][1] )

songList.sample(n=10)
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import seaborn as sns

customStopWords = ["'s", "n't", "'m", "'re", "'ll","'ve","...", "ä±", "''", '``',\
                  '--', "'d", 'el', 'la']
stopWords = stopwords.words('english') + customStopWords

words = ""
for song in songList.iterrows():
    words += " " + song[1]['lyrics']

words = nltk.word_tokenize( words.lower() )
words = [ word for word in words if len(word) > 1\
                             and not word.isnumeric()\
                             and word not in stopWords ]
    
word_dist = FreqDist( words  )
print("The 10 most common words in the dataset are :")
for word, frequency in word_dist.most_common(10):
    print( u'{} : {}'.format( word, frequency ) )

plt.figure(figsize=(15, 10))
nlp_words = word_dist.plot( 20 )
for genre in songList['genre'].unique():
    dfGenre = songList[songList['genre']==genre]
    #print (genre + " : " + str(len(dfGenre)) + " songs.") 
    words = ""
    for song in dfGenre.iterrows():
        words += " " + song[1]['lyrics']
        
    words = nltk.word_tokenize( words.lower() )
    words = [ word for word in words if len(word) > 1\
                             and not word.isnumeric()\
                             and word not in stopWords ]
    word_dist = FreqDist( words )
    #for word, frequency in word_dist.most_common(5):
    #    print(u'{} : {}'.format(word, frequency))
    plt.title( genre )
    lp_words = word_dist.plot( 20 )
test_sample = songList.sample(5)
for song in test_sample.iterrows():
    words = nltk.word_tokenize( song[1]['lyrics'].lower() )
    words = [ word for word in words if len(word) > 1\
                             and not word.isnumeric()\
                             and word not in stopWords ]
    
    word_dist = FreqDist( words )
    plt.title( song[1]['song'] + " - " + str(song[1]['genre']) )
    word_dist.plot( 20 )
word_dist = []
for song in songList.iterrows():
    words = nltk.word_tokenize( song[1]['lyrics'].lower() )
    words = [ word for word in words if len(word) > 1\
                             and not word.isnumeric()\
                             and word not in stopWords ]
    word_dist.append( FreqDist( words ) )
songList['word_dist'] = word_dist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB



train, test = train_test_split(songList, test_size=0.2)

X_train, y_test  = train[['song', 'year', 'artist']], train['genre']
X_test,  y_train = test[['song', 'year', 'artist']], test['genre']
X_train, y_test  = train[ 'year' ], train['genre']
X_test,  y_train = test[ 'year' ], test['genre']


clf = MultinomialNB().fit( X_train, y_train )
clf = SVC()
#clf.set_params(kernel='linear').fit(X_train, y_train)
