# Import general purpose packages
import numpy as np 
import pandas as pd 
import sys
import json
import os
import re

# This is what we are using for data preparation and ML part (thanks, Rafal, for great tutorial)
from sklearn.feature_extraction import text      
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
# Different ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

#Global parameters
SONGS_PER_GENRE = 10000
SONGS_PER_TRAINING = 1000
SONGS_PER_TESTING = 100

#we need to cleanse lyrics to remove special characters and garbage
#this function cleanses single string
def cleanse (text):
    result = re.sub('[^a-zA-Z0-9 ]', ' ', text)
    return result


partists = pd.read_csv('/kaggle/input/scrapped-lyrics-from-6-genres/artists-data.csv') #load the list of artists
psongs = pd.read_csv('/kaggle/input/scrapped-lyrics-from-6-genres/lyrics-data.csv') #load the list of songs


pop_artists = partists[partists['Genre']=='Pop'] # filter artists by genre
pop_songs = pd.merge(psongs, pop_artists, how='inner', left_on='ALink', right_on='Link') #inner join of pop artists with songs to get only songs by pop artists
pop_songs = pop_songs[['Genre', 'Artist', 'SName', 'Lyric']].rename(columns={'SName':'Song'})#leave only columns of interest and rename some of them.
pop_songs = pop_songs.dropna() # Remove incomplete records, cleanse lyrics
pop_songs = pop_songs[pop_songs['Lyric']!='Instrumental'].head(SONGS_PER_GENRE).applymap(cleanse) #Remove instrumental compositions  and limit the size of final dataset
pop_songs.head()
#raw data
songs = pd.read_csv('/kaggle/input/150k-lyrics-labeled-with-spotify-valence/labeled_lyrics_cleaned.csv') #load the list of songs
artists = pd.read_csv('/kaggle/input/ultimate-spotify-tracks-db/SpotifyFeatures.csv') #load artists by genere spotify database
rap_artists=artists[artists['genre']=='Rap'][['genre','artist_name']].drop_duplicates() #extract only artist names and genres, deduplicate the list
rap_songs = pd.merge(songs, rap_artists, how='inner', left_on='artist', right_on='artist_name') #inner join of rap artists with songs to get only songs by rap artists
rap_songs = rap_songs[['genre', 'artist', 'song', 'seq']].rename(columns={'genre':'Genre', 'artist':'Artist', 'song':'Song','seq':'Lyric'}) #leave only columns of interest and rename some
rap_songs = rap_songs.dropna().head(SONGS_PER_GENRE).applymap(cleanse) #remove incomplete records and limit dataset size
rap_songs.head()
# since I don't need 250K songs for my analysis and I don't want to wait forever for data to load 
# I am going to build a little function, which is going to return required number of songs
def list_n_metal_songs (limit):
    # intputs:
    # limit - number of songs to return
    # outputs:
    # the list of file paths to songs of required length
    counter = 1
    result = []
    # lets repurpose kaggle boilerplate code to obtain the list of all files with lyrics
    for dirname, _, filenames in os.walk('/kaggle/input/large-metal-lyrics-archive-228k-songs/metal_lyrics'):
        for filename in filenames:
            result.append(os.path.join(dirname, filename))
            #some really bad programming pattern here
            if counter >= limit: return result
            else: counter+=1

# let's get limited set of songs                
metal_files = list_n_metal_songs(SONGS_PER_GENRE)

# let's build our dataset 
artist = []
song = []
lyric = []

#we are going to iterate through the list of files
for path in metal_files:
    # now, since the format of the path is known: <letter>/<artist>/<album>/<track #>.<track name>.txt we can obtain all metadata from file path
    p = path.split('/')
    s = p[8].split('.')
    artist.append(p[6])
    song.append(s[1].strip())
    #we can also open and read file content
    f = open (path, 'r')
    lyric.append(f.read())
    f.close()

#finally, we can assemble metal songs dataset
metal_songs = pd.DataFrame({'Genre':'Metal', 'Artist' : artist, 'Song' : song,'Lyric' : lyric}).applymap(cleanse)
metal_songs.head()
print("Pop -",len(pop_songs), "| Rap -", len(rap_songs), "| Metal -",len(metal_songs))
#prepare training dataset by taking equal scoops from each dataset
training_data = pd.concat([pop_songs.head(SONGS_PER_TRAINING), rap_songs.head(SONGS_PER_TRAINING), metal_songs.head(SONGS_PER_TRAINING)])
#Creating vocabulary
cv = CountVectorizer(strip_accents='ascii', lowercase=True, stop_words='english', analyzer='word') #create the CountVectorizer object
cv.fit(training_data['Lyric'].values) #fit into our dataset

#Creating bag of words representations
bow = cv.transform(training_data['Lyric'].values) 
print(bow.shape[0], 'samples x ',bow.shape[1],'words in vocabulary' )
#Create the machine learning classifier object
models = {'Logistic Regression':LogisticRegression(max_iter=500), 'Linear SVC':LinearSVC(max_iter=10000),'Decision Tree': DecisionTreeClassifier(), 'Gradient Descent': SGDClassifier()}
# things that didn't work 'Nu SVC': NuSVC(), 'SVC': SVC(), 'Naive Bayes': GaussianNB()
for the_model in models.keys():
    print ("Training", the_model)
    models[the_model].fit(bow.toarray(), training_data['Genre'])

#prepare data for testing
test_data=pd.concat([pop_songs.iloc[SONGS_PER_TRAINING:SONGS_PER_TRAINING+SONGS_PER_TESTING], rap_songs.iloc[SONGS_PER_TRAINING:SONGS_PER_TRAINING+SONGS_PER_TESTING], metal_songs.iloc[SONGS_PER_TRAINING:SONGS_PER_TRAINING+SONGS_PER_TESTING]])

#convert lyrics to bag of word representation
test_bow = cv.transform(test_data['Lyric'].values) 
#predict labels for all the sentences
accuracy = {}
pg = test_data[['Artist', 'Song','Genre']]
for the_model in models.keys():
    print("Evaluating",the_model)
    pred_genre= models[the_model].predict(test_bow.toarray())
    pg[the_model]=pred_genre
    accuracy[the_model]=accuracy_score(test_data['Genre'],pred_genre)
print ("---Accuracy Scores---")
print(accuracy)
#get user input
user_input = input()

#transform it to bag of words
user_bow = cv.transform([cleanse(user_input)]) 

#run against trained models
for the_model in models.keys():
    print(the_model, models[the_model].predict(user_bow))
