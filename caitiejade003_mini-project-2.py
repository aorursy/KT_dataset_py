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
#I initially thought I may need to have the artists in separate datasets

import pandas as pd

Doorsdown = pd.read_csv("../input/lyrics3artists/3-doors down.csv")

Adele = pd.read_csv("../input/lyrics3artists/Adele.csv")

Beyonce = pd.read_csv("../input/lyrics3artists/Beyonce.csv")

#Just making sure I can grab the specific lyrics



Beyonce.iloc[3]['lyrics']
#make a text file with all the lyrics from Beyonce

with open("beyoncelyrics.txt", "w") as b_lyrics:

    for ind in Beyonce.index: 

        lyric = str(Beyonce['lyrics'][ind])

        lyric = lyric.replace('\n',' ')

        b_lyrics.write(lyric)
#making a bag of words dictionary that assigns an id to each word from beyonce's lyrics

from sklearn.feature_extraction.text import CountVectorizer

# list of text documents

text = open("beyoncelyrics.txt", "r")

# create the transform

vectorizer = CountVectorizer()

# tokenize and build vocab

vectorizer.fit(text)

# summarize

b_vocab = vectorizer.vocabulary_

print(b_vocab)

# encode document

vector = vectorizer.transform(text)

# summarize encoded vector

print(vector.shape) #this shows how many words in the dictionary

print(type(vector)) #shows type of matrix

print(vector.toarray()) #supposed to show counts of the words, but appears blank

print(b_vocab['heart'])
#feature extraction https://scikit-learn.org/stable/modules/feature_extraction.html

from sklearn.feature_extraction import DictVectorizer

#measurements = [vector]

vec = DictVectorizer()



vec.fit_transform(b_vocab).toarray()



vec.get_feature_names()  #this is all the words in my new vocabulary of beyonce lyrics
#trying another example

text2 = ["our love forever dynasty"]

vector = vectorizer.transform(text2)

print(vector.toarray()) #there's too many items to see where those words are in the dictionary
#making a dataframe and replacing new lines with spaces in the lyrics column

import pandas as pd

df = pd.read_csv("../input/allmusic/FiveArtists.csv")

df = df.replace(r'\n',' ', regex=True) 



df.head()

#example pulled from tutorial

from io import StringIO

col = ['artist', 'lyrics']  #isolating the columns needed for the model

df = df[col]

df = df[pd.notnull(df['lyrics'])] #ensuring there are not null values

df.columns = ['artist', 'lyrics']

df['category_id'] = df['artist'].factorize()[0]  

category_id_df = df[['artist', 'category_id']].drop_duplicates().sort_values('category_id') #creates an id for the artist column and turns it into a new category_id column

category_to_id = dict(category_id_df.values)

id_to_category = dict(category_id_df[['category_id', 'artist']].values)

df.head(25)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.lyrics).toarray()

labels = df.category_id

features.shape  #there are 101 lyrics represented by 248 features, that seems like too few features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.lyrics).toarray()

labels = df.category_id

features.shape  #reduced the min_df because I expect fewer repeat words and wanted to increase the amount of features
from sklearn.feature_selection import chi2

import numpy as np

N = 2

for artist, category_id in sorted(category_to_id.items()):

  features_chi2 = chi2(features, labels == category_id)

  indices = np.argsort(features_chi2[0])

  feature_names = np.array(tfidf.get_feature_names())[indices]

  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

  print("# '{}':".format(artist))

  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))

  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    

#this has identified most common single-word and two-word features for each artist
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['lyrics'], df['artist'], random_state = 0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)



X_train_counts.shape
print(clf.predict(count_vect.transform(["In the darkness of my heart is a shadow of your face From the deepest part of my regret I hear the words I wish I'd said At the dawning of the day"])))
#sample from 3 doors down doesn't work

print(clf.predict(count_vect.transform(["She walks through the city No one recognizes her face They don't want her pity No one ever mentions her name She's carried the broken Their scars have no name in her heart She walks in forgiveness She'll shine like a light in the dark She is love She is love"])))
#sample from beyonce - works!

print(clf.predict(count_vect.transform(["Come take my hand I won't let you go I'll be your friend I will love you so, deeply I will be the one to kiss you at night I will love you until the end of time I'll be your baby And I promise not to let you go Love you like crazy Now say you'll never let me go Say you'll never let me go Say you'll never let me go Take, you away, from here It's nothing between us but space, and time I'll be your own little star, let me shine you up Or your own little universe make me your girl I will be the one to kiss you at night I will love you until the end of time I'll be your baby And I promise not to let you go Love you like crazy Now say you'll never let me go Say you'll never let me go (say) Say you'll never let me go Say you'll never let me go (say) Say you'll never let me go"])))
#Next steps

#create a dataset of non-trained songs predict the artist and compare

    #make separate csv of other songs -done

    #make new dataframe -done

    #go through the lyrics in the dataframe and predict which artist it is - done

    #add that artist name to a new column in the dataframe - done

    #compare the column with the actual artist to the predicted one  - done

    #make a dictionary with artist name and the number of correct predictions - done

#calculate #true for each artist and make a visual to show which ones are most accurate

#create larger training sets and see if it's more accurate
testdata = pd.read_csv("../input/sameartistcompare/comparisonsongs.csv")

testdata = testdata.replace(r'\n',' ', regex=True) 

#songlyric = songlyric.replace('\n',' ')  #removing new line characters

testdata['prediction'] = "" 

testdata.head()
for ind in testdata.index: #iterating through test data

        songlyric = str(testdata['lyrics'][ind]) #making variable from the lyrics column      

        prediction = clf.predict(count_vect.transform([songlyric]))  #assigning a variable to the prediction

        testdata['prediction'][ind] = prediction #adding that to the dataframe

       



        



testdata



#print(clf.predict(count_vect.transform([songlyric]))) #printing the prediction
#compare the artists and prediciton columns to find out the number of right predictions by artist and total

correct = 0

incorrect = 0



mydict = {}

for ind in testdata.index:

    realartist = str(testdata['artist'][ind])

    prediction = testdata['prediction'][ind]

    prediction = str(prediction[0])

    if realartist == prediction:

        correct = correct + 1

        if realartist in mydict:    #check if realartist is currently a key in the dictionary, if not create it and set value to 1. If it is get the current value and increment by 1.

            mydict[realartist] += 1

        else:

            mydict[realartist] = 1

    else:

        incorrect = incorrect + 1



print(mydict)





percentcorrect = correct/(correct + incorrect) * 100        

print('This model predicts artists correctly %d percent of the time overall.' %percentcorrect)
#I updated the code above so it would include any artists with 0 correct predictions i.e. Coldplay

correct = 0 

incorrect =0



mydict = {}

for ind in testdata.index:

    realartist = str(testdata['artist'][ind])

    prediction = testdata['prediction'][ind]

    prediction = str(prediction[0])

    if realartist not in mydict: 

        mydict[realartist] = 0

    if realartist == prediction:

        correct = correct + 1

        mydict[realartist] += 1                  

    else:

        incorrect = incorrect + 1



print(mydict)





percentcorrect = correct/(correct + incorrect) * 100        

print('This model predicts artists correctly %d percent of the time overall.' %percentcorrect)
#This plot shows how many of each artist's predictions were correct. Beyonce's were most accurate while coldplay was completely wrong!

import matplotlib.pyplot as plt



D = {u'Label1':26, u'Label2': 17, u'Label3':30}



plt.bar(range(len(mydict)), list(mydict.values()), align='center')

plt.xticks(range(len(mydict)), list(mydict.keys()), rotation=20)

Lyricsdf = pd.read_csv("../input/moresongs/FiveArtists2.csv")

Lyricsdf = Lyricsdf.replace(r'\n',' ', regex=True) 



Lyricsdf

col = ['artist', 'lyrics']  #isolating the columns needed for the model

Lyricsdf = Lyricsdf[col]

Lyricsdf.columns = ['artist', 'lyrics']

Lyricsdf['category_id'] = Lyricsdf['artist'].factorize()[0]  

category_id_df = Lyricsdf[['artist', 'category_id']].drop_duplicates().sort_values('category_id') #creates an id for the artist column and turns it into a new category_id column

category_to_id = dict(category_id_df.values)

id_to_category = dict(category_id_df[['category_id', 'artist']].values)

Lyricsdf.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(Lyricsdf.lyrics).toarray()

labels = Lyricsdf.category_id

features.shape  #increased the min_df back up to see if its actually better to have less features - this makes 643
N = 2

for artist, category_id in sorted(category_to_id.items()):

  features_chi2 = chi2(features, labels == category_id)

  indices = np.argsort(features_chi2[0])

  feature_names = np.array(tfidf.get_feature_names())[indices]

  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

  print("# '{}':".format(artist))

  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))

  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    

#here are the most correlated unigrams and bigrams...seems like anything with an apostrophe gets cut off
#I realized from my original code above that it may have split up the data, so I am going to try to train the model without splitting it up since I have a separate test dataset already

X_train = Lyricsdf['lyrics']

y_train = Lyricsdf['artist']

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)



#>>> tfidf_transformer = TfidfTransformer()

#>>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#>>> X_train_tfidf.shape

#clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)



X_train_counts.shape
#retesting- correct prediction of Adele

print(clf.predict(count_vect.transform(["In the darkness of my heart is a shadow of your face From the deepest part of my regret I hear the words I wish I'd said At the dawning of the day"])))
#retesting- correct prediction of Three Doors Down

print(clf.predict(count_vect.transform(["She walks through the city No one recognizes her face They don't want her pity No one ever mentions her name She's carried the broken Their scars have no name in her heart She walks in forgiveness She'll shine like"])))
#seeing if I can use the same dataframe as before....

testdata['prediction2'] = "" 

testdata.head()
for ind in testdata.index: #iterating through test data

        songlyric = str(testdata['lyrics'][ind]) #making variable from the lyrics column      

        newprediction = clf.predict(count_vect.transform([songlyric]))  #assigning a variable to the prediction

        testdata['prediction2'][ind] = newprediction #adding that to the dataframe

       



        



testdata
#I updated the code above so it would include any artists with 0 correct predictions i.e. Coldplay

correct2 = 0 

incorrect2 =0



mydict2 = {}

for ind in testdata.index:

    realartist = str(testdata['artist'][ind])

    newprediction = testdata['prediction2'][ind]

    newprediction = str(newprediction[0])

    if realartist not in mydict2: 

        mydict2[realartist] = 0

    if realartist == newprediction:

        correct2 = correct + 1

        mydict2[realartist] += 1                  

    else:

        incorrect2 = incorrect + 1



print(mydict2)





percentcorrect = correct/(correct + incorrect) * 100        

print('This model predicts artists correctly %d percent of the time overall.' %percentcorrect)



#Three doors down is now perfect, Colplay is the same total fail, Adele is slightly better, Beyonce much worse and Dido slightly worse.
#This plot shows how many of each artist's predictions were correct. 

D = {u'Label1':26, u'Label2': 17, u'Label3':30}



plt.bar(range(len(mydict2)), list(mydict2.values()), align='center')

plt.xticks(range(len(mydict2)), list(mydict2.keys()), rotation=20)