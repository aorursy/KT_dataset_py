import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib

import json

import nltk

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten, Dropout

from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten, Dropout, Bidirectional

from keras.layers import LSTM, Dropout,Activation, Bidirectional



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from nltk.stem.porter import *

import random

import copy
def removeStopWords(lowerArg):

    i=0

    removed=[]

    for x in lowerArg:

        i+=1

        removed.append((' '.join([word for word in x.split() if word not in nltk.corpus.stopwords.words('english')])))

        

    return pd.Series(removed).astype(str)
df=pd.read_csv('../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')

newsDF=pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True,dtype='str')
##Limit to just comedies and Genres

genres=['drama','comedy']

df=df[df['Genre'].isin(genres)]

df=df.reset_index()

df['GenreID']=df['Genre'].apply(lambda x: genres.index(x))



wordCount=df['Plot'].apply(lambda x: x.count(' '))

print("Mean number of words in synopses: ",int(wordCount.mean()))

print("Standard deviation number of words in synopses: ", int(wordCount.std()))

print('Number of Dramas: ',df['Genre'].value_counts()[0])

print('Number of Comedies: ',df['Genre'].value_counts()[1])

matplotlib.pyplot.hist(wordCount)

print('Distribution of Synopsis Word Counts')
synNumber=random.randint(1,1000)

print(df['Title'].loc[synNumber])

print(df['Genre'].loc[synNumber])

print(df['Plot'].loc[synNumber])
##Take roughly balanced sample of news dataset between comedy and non-comedy stories

newsDF['Comedy']=(newsDF['category']=='COMEDY')

news=pd.concat([newsDF[newsDF['Comedy']==False].sample(5000),newsDF[newsDF['Comedy']]],axis=0)

print('News Comedies: ',news['Comedy'].value_counts().values[0])

print('News Non-Comedies: ',news['Comedy'].value_counts().values[1])



wordCountNews=newsDF['short_description'].apply(lambda x: x.count(' '))

wordCountHeadline=newsDF['headline'].apply(lambda x: x.count(' '))



print("Mean number of words in synopsis: ",int(wordCountNews.mean()))

print("Standard deviation number of words in synopsis: ", int(wordCountNews.std()))

print()

print("Mean number of words in headline: ",int(wordCountHeadline.mean()))

print("Standard deviation number of words in headline: ", int(wordCountHeadline.std()))
news=news.reset_index()

news['Text']=news['short_description'].str.cat(news['headline'])

news['OriginalText']=news['Text']

from nltk.stem.porter import *

stemmer = PorterStemmer()

df['StemmedPlot']=df['Plot'].str.split().apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))

lower=news['Text'].str.lower()

noStops=removeStopWords(lower)

news['Text']=noStops



#Store Original Plot for later

df['OriginalPlot']=df['Plot']



lower=df['Plot'].str.lower()

cleaned=removeStopWords(lower)

df['Plot']=cleaned
##Using non-stemmed

tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(df['Plot']))

sequences = tokenizer.texts_to_sequences(list(df['Plot']))

maxLen=np.max([len(sequence) for sequence in sequences])

print("Maximum Length: ",maxLen)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxLen)



#labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)

#print('Shape of label tensor:', labels.shape)



newstokenizer = Tokenizer()

newstokenizer.fit_on_texts(list(news['Text']))

newsSequences = newstokenizer.texts_to_sequences(list(news['Text']))

newsword_index = newstokenizer.word_index

print('Found %s unique tokens.' % len(newsword_index))

newsdata = pad_sequences(newsSequences, maxlen=maxLen)



#labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', newsdata.shape)

#print('Shape of label tensor:', labels.shape)
##Sanity Check index is the word_index dictionary with keys reversed

sanityCheckIndex={v: k for k, v in tokenizer.word_index.items()}

print(sequences[500])

print(' '.join([sanityCheckIndex[wordIndex] for wordIndex in sequences[500]]))

print(data[500][0])

print(data[500][-1])

print(' '.join([sanityCheckIndex[wordIndex] for wordIndex in data[500] if wordIndex!=0 ]))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

tfs = tfidf.fit_transform(df['StemmedPlot'])

print('Shape of TF-IDF matrix: ', tfs.T.shape)



seed=random.randint(1,1000)

X_train, X_test, y_train, y_test = train_test_split(data, df['GenreID'], test_size=0.2, random_state=seed)

testIndices=y_test.index

y_train=to_categorical(y_train)

y_test=to_categorical(y_test)



y_train_small=y_train.copy()

X_train_small=X_train.copy()



y_train_add=to_categorical(news['Comedy'])

X_train_add=newsdata

X_train=np.concatenate([X_train,X_train_add],axis=0)

y_train=np.concatenate([y_train, y_train_add],axis=0)

print("X Train without news shape: ",X_train_small.shape)

print("Y train without news shape: ",y_train_small.shape)

print("X train with news shape: ",X_train.shape)

print("Y train with news shape: ",y_train.shape)

print("X test shape: ",X_test.shape)

print("Y test shape: ",y_test.shape)

X_trainBag, X_testBag, y_trainBag, y_testBag = train_test_split(tfs, df['GenreID'], test_size=0.2, random_state=seed)

testIndicesBag=y_testBag.index

y_trainBag=to_categorical(y_trainBag)

y_testBag=to_categorical(y_testBag)

print("BoW X Train Shape: ",X_trainBag.shape)

print("BoW Y Train Shape: ",y_trainBag.shape)

print("BoW X Test Shape: ",X_testBag.shape)

print("BoW Y Test Shape: ",y_testBag.shape)

from keras import *

from keras.layers import Dense

from keras.utils import to_categorical

tf_input = Input(shape=(tfs.shape[1],), dtype='float32')

x=Dense(len(genres),activation='sigmoid')(tf_input)

bagOfWords = Model(tf_input, x)

bagOfWords.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])



bagOfWords.fit(X_trainBag, y_trainBag, validation_data=(X_testBag, y_testBag),epochs=20, batch_size=128)

embeddings={}

index=0

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt') as file:

    for embeddingLine in file:

        lineSplit=embeddingLine.split()

        coefs = np.asarray(lineSplit[1:], dtype='float32')

        embeddings[lineSplit[0]]=coefs

        index+=1



embeddings_matrix=np.zeros((len(word_index)+1,len(embeddings['a'])))

for word,i in word_index.items():

    if word in embeddings:

        embeddings_matrix[i]=embeddings[word]



print('Word #2: ',sanityCheckIndex[2])

print('Index of him : ',word_index['him'])

print('Embbedding in embeddings list: ',embeddings['him'][:5])

print('Embedding in embeddings matrix: ',embeddings_matrix[2][:5])

from keras.layers import Embedding



embedding_layer = Embedding(len(word_index) + 1,

                            len(embeddings['a']),

                            weights=[embeddings_matrix],

                            input_length=maxLen,

                            trainable=False)

embedding_layerNoGlove = Embedding(len(word_index) + 1,

                            len(embeddings['a']),

                            weights=[embeddings_matrix],

                                   input_length=maxLen,

                            )

sequence_input = Input(shape=(maxLen,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

embeddingOnlyModel = Model(sequence_input, embedded_sequences)



print('Manual Embeddings Result: ',[list(embeddings[sanityCheckIndex[x]][:3]) if sanityCheckIndex[x] in embeddings else [0,0,0] for x in sequences[500] ][-5:])

##print(sequences[500])

##print([ sanityCheckIndex[l] for l in list(data[500]) if l>0 ])

##print([ sanityCheckIndex[l] for l in list(sequences[500]) if l>0 ])

print('Model Embeddings Result: ',embeddingOnlyModel.predict(np.array(data[500]).reshape(1,maxLen))[0,-5:,:3])

##print(embeddings_matrix[2][:5])

sequence_input = Input(shape=(maxLen,), dtype='int32')

embedded_sequences = embedding_layerNoGlove(sequence_input)

x=Conv1D(128, 9, activation='relu')(embedded_sequences)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)

x = Flatten()(x)

x=Dense(128, activation='relu')(x)

x=Dense(len(genres),activation='softmax')(x)



noGloveCNN = Model(sequence_input, x)

noGloveCNN.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])

noGloveCNN.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=2, batch_size=128)
from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten, Dropout, Bidirectional

sequence_input = Input(shape=(maxLen,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x=Conv1D(128, 9, activation='relu')(embedded_sequences)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)



x = Flatten()(x)

x=Dense(128, activation='relu')(x)

x=Dense(len(genres),activation='softmax')(x)



model = Model(sequence_input, x)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])

X_train.shape

model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=2, batch_size=128)


sequence_input = Input(shape=(maxLen,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x=Conv1D(128, 9, activation='relu')(embedded_sequences)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)



x = Flatten()(x)

x=Dense(128, activation='relu')(x)

x=Dense(len(genres),activation='softmax')(x)



modelSmall = Model(sequence_input, x)

modelSmall.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])

X_train.shape

modelSmall.fit(X_train_small, y_train_small, validation_data=(X_test, y_test),epochs=2, batch_size=128)


from keras.layers import LSTM, Dropout,Activation, Bidirectional

word_indices =Input(shape=(maxLen,), dtype='int32')

# Propagate sentence_indices through your embedding layer, you get back the embeddings

embeddingsLSTM = embedding_layer(word_indices)   



# Propagate the embeddings through an LSTM layer with 128-dimensional hidden state

# Be careful, the returned output should be a batch of sequences.

x=Conv1D(128, 9, activation='relu')(embeddingsLSTM)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)



X =  LSTM(128,return_sequences=False)(x)

# Add dropout with a probability of 0.5

X = Dropout(.65)(X)

# Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.

X = Dense(len(genres),activation='softmax')(X)

# Add a softmax activation

X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.

LSTMmodel = Model(inputs = word_indices, outputs = X) 

LSTMmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

LSTMmodel.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=2, batch_size=128)
from keras.layers import LSTM, Dropout,Activation, Bidirectional

word_indices =Input(shape=(maxLen,), dtype='int32')

# Propagate sentence_indices through your embedding layer, you get back the embeddings

embeddingsLSTM = embedding_layer(word_indices)   



# Propagate the embeddings through an LSTM layer with 128-dimensional hidden state

# Be careful, the returned output should be a batch of sequences.

x=Conv1D(128, 9, activation='relu')(embeddingsLSTM)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)

x=Conv1D(128, 9, activation='relu')(x)

x = Dropout(.4)(x)

x=MaxPooling1D(9)(x)



X =  LSTM(128,return_sequences=False)(x)

# Add dropout with a probability of 0.5

X = Dropout(.65)(X)

# Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.

X = Dense(len(genres),activation='softmax')(X)

# Add a softmax activation

X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.

LSTMmodelSmall = Model(inputs = word_indices, outputs = X) 

LSTMmodelSmall.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

LSTMmodelSmall.fit(X_train_small, y_train_small, validation_data=(X_test, y_test),epochs=5, batch_size=128)
CNNpreds=model.predict(X_test)
LSTMpreds=LSTMmodel.predict(X_test)
noGloveCNNpreds=noGloveCNN.predict(X_test)
bagPreds=bagOfWords.predict(X_testBag)

avgPreds=np.average([bagPreds,LSTMpreds,CNNpreds],weights=[.8,.2,.2],axis=0)

avgPreds=bagPreds



withPreds=pd.concat([pd.DataFrame(avgPreds),df.loc[testIndices,['OriginalPlot','Genre','Title']].reset_index()],axis=1)

withPreds['Predicted Genre']=(withPreds[0]>withPreds[1]).replace(True,'drama').replace(False,'comedy')

accuracy=(withPreds['Predicted Genre']==withPreds['Genre']).mean()

print('Accuracy of final ensembled model: ',accuracy)
pd.pivot_table(withPreds,columns='Predicted Genre', index='Genre', aggfunc=len)['index']

for x in withPreds.sort_values(by=0)[['Title','OriginalPlot',0,'Genre']].head(3).iterrows():

    print(x[1].Title,x[1][0],x[1]['Genre'])

    print(x[1].OriginalPlot)

    print(' ')
for x in withPreds.sort_values(by=1)[['Title','OriginalPlot',0,'Genre']].head(3).iterrows():

    print(x[1].Title,x[1][0],x[1]['Genre'])

    print(x[1].OriginalPlot)

    print(' ')
weights=bagOfWords.get_weights()[0][:,0]

mostDramatic=weights.argsort()[-10:][::-1]

leastDramatic=weights.argsort()[:10][::1]

index_to_words={v: k for k, v in tfidf.vocabulary_.items()}
print('Words most likely to indicate comedy: ', [index_to_words[x] for x in leastDramatic])
print('Words most likely to indicate drama: ',[index_to_words[x] for x in mostDramatic])
def bowPredict(syn):

    noStops=removeStopWords([syn])

    stemmed=' '.join([stemmer.stem(x) for x in noStops[0].split() ])

    mat=tfidf.transform([stemmed])

    preds=bagOfWords.predict(mat)

    if preds[0][0]>preds[0][1]: return 'comedy'

    return 'drama'
bowPredict("United States Naval Aviator LT Pete ‘Maverick’ Mitchell and his Radar Intercept Officer LTJG Nick ‘Goose’ Bradshaw fly the F-14A Tomcat aboard USS Enterprise (CVN-65). During an interception with two hostile MiG-28aircraft (portrayed by a Northrop F-5), Maverick gets missile lock on one, while the other hostile aircraft locks onto Maverick's wingman, Cougar. While Maverick drives off the remaining MiG-28, Cougar is too shaken to land, and Maverick, defying orders, shepherds him back to the carrier. Cougar gives up his wings, citing his newborn child that he has never seen. Despite his dislike for Maverick's recklessness, CAG ’Stinger’ sends him and Goose to attend Topgun,[6] the Naval Fighter Weapons School at Naval Air Station Miramar.At a bar the day before Topgun starts, Maverick, assisted by Goose, unsuccessfully approaches a woman. He learns the next day that she is Charlotte ‘Charlie’ Blackwood, an astrophysicist and civilian Topgun instructor. She becomes interested in Maverick upon learning of his inverted maneuver with the MiG-28, which disproves US intelligence on the enemy aircraft's performance. During Maverick's first training sortie he defeats instructor LCDR Rick ‘Jester’ Heatherly but through reckless flying breaks two rules of engagement and is reprimanded by chief instructor CDR Mike ‘Viper’ Metcalf. Maverick also becomes a rival to top student LT Tom ‘Iceman’ Kazansky, who considers Maverick's flying ‘dangerous.’ Charlie also objects to Maverick's aggressive tactics but eventually admits that she admires his flying and omitted it from her reports to hide her feelings for him, and the two begin a romantic relationship. During a training sortie, Maverick abandons his wingman ‘Hollywood’ to chase Viper, but is defeated when Viper maneuvers Maverick into a position from which his wingman Jester can shoot down Maverick from behind, demonstrating the value of teamwork over individual prowess. Maverick and Iceman, now direct competitors for the Topgun Trophy, chase an A-4 in a later training engagement. Maverick pressures Iceman to break off his engagement so he can shoot it down, but Maverick's F-14 flies through the jet wash of Iceman's aircraft and suffers a flameout of both engines, going into an unrecoverable flat spin. Maverick and Goose eject, but Goose hits the jettisoned aircraft canopy head-first and is killed. Although the board of inquiry clears Maverick of responsibility for Goose's death, he is overcome by guilt and his flying skill diminishes. Charlie and others attempt to console him, but Maverick considers retiring. He seeks advice from Viper, who reveals that he served with Maverick's father Duke Mitchell on the USS Oriskany and was in the air battle in which Mitchell was killed. Contrary to official reports which faulted Mitchell, Viper reveals classified information that proves Mitchell died heroically, and informs Maverick that he can succeed if he can regain his self-confidence. Maverick chooses to graduate, though Iceman wins the Top Gun Trophy. During the graduation party, Viper calls in the newly graduated aviators with the orders to deploy. Iceman, Hollywood, and Maverick are ordered to immediately return to Enterprise to deal with a ‘crisis situation’, providing air support for the rescue of a stricken ship that has drifted into hostile waters. Maverick and Merlin (Cougar's former RIO) are assigned as back-up for F-14s flown by Iceman and Hollywood, despite Iceman's reservations over Maverick's state of mind. The subsequent hostile engagement with six MiGs sees Hollywood shot down; Maverick is scrambled alone due to a catapult failure and nearly retreats after encountering circumstances similar to those that caused Goose's death. Upon finally rejoining Iceman, Maverick shoots down three MiGs, and Iceman one, forcing the other two to flee. Upon their triumphant return to Enterprise, Iceman and Maverick express newfound respect for each other. Offered any assignment he chooses, Maverick decides to return to Topgun as an instructor. At a bar in Miramar, Maverick and Charlie reunite.")
