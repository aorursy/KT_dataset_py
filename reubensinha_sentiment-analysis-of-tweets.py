#Straight forward neural network
import numpy as np 
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import os

def ingest():
    data = pd.read_csv('../input/tweets-16m/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", header = None)
    data.drop([1,2,3,4], axis=1, inplace=True)
    data.columns = ['sentiment', 'text']
    data = data[data.sentiment.isnull() == False]
    data = data[data['text'].isnull() == False]
    data.sentiment = data.sentiment.apply(lambda x: 'positive' if x == 4 else 'negative')
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape)    
    return data

data = ingest()
vectorizer = TfidfVectorizer(ngram_range = (1,2))
tfidf_word_freq = vectorizer.fit_transform(tqdm(data.text))
input_data = tfidf_word_freq
n_features = input_data.shape[1]
target = pd.get_dummies(data.sentiment)
#NN model
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (n_features,)))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
early_stopping = EarlyStopping(patience = 3)
model.fit(input_data, target, callbacks = [early_stopping], validation_split = 0.2)
#Save model
model.save('Sentiment_analysis_module.HD5')
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from gensim.models.doc2vec import LabeledSentence

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#Straight forward neural network
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import os

data1 = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')[['airline_sentiment', 'text']]
data1.columns = ['sentiment', 'text']
data2 = pd.read_csv('../input/twitter-sentiment-analysis-with-textblob/twitter_data_2.csv')[['Tweet_text', 'Class']]
data2.columns = ['text', 'sentiment']
data2.sentiment = data2.sentiment.apply(lambda x: x.lower())
data = pd.concat([data1, data2])

vectorizer = TfidfVectorizer(ngram_range = (1,2))
tfidf_word_freq = vectorizer.fit_transform(data.text)
input_data = tfidf_word_freq
target = pd.get_dummies(data.sentiment)
#NN model
model = Sequential()
model.add(Dense(300, activation = 'relu', input_shape = (172685,)))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(300, activation = 'relu'))
#model.add(Dense(200, activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(Dense(130, activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(Dense(130, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
early_stopping = EarlyStopping(patience = 3)
#model.fit(input_data, target, callbacks = [early_stopping])
#Save model
#model.save('Sentiment_analysis_module_2.HD5')
data.shape
#Straight forward Neural network prediction on test set
import os
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from sqlalchemy import create_engine
data1 = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')[['airline_sentiment', 'text']]
data1.columns = ['sentiment', 'text']
data2 = pd.read_csv('../input/twitter-sentiment-analysis-with-textblob/twitter_data_2.csv')[['Tweet_text', 'Class']]
data2.columns = ['text', 'sentiment']
data2.sentiment = data2.sentiment.apply(lambda x: x.lower())
data = pd.concat([data1, data2])
data_test = pd.read_csv('../input/twitter-sentiment-analysis/twitter_data_1.csv')
#Vectorizer
vectorizer = TfidfVectorizer(ngram_range = (1,2))
tfidf_word_freq = vectorizer.fit(data.text)
input_data = vectorizer.transform(data_test.dropna().Tweet_text)
model = keras.models.load_model('Sentiment_analysis_module_2.HD5')
preds = model.predict(input_data)
def get_index(row):
    x = max(row)
    if x == row[0]:
        return 0;
    if x == row[1]:
        return 1;
    return 2

result_set = ['negative', 'neutral', 'positive']
count = 0
for row in range(len(preds)):
    if (result_set[get_index(preds[row])] == data_test.Class[row].lower()):
        count+=1
print('Accuracy on test set using NN: ', count/len(preds))
#Straight forward Neural network prediction on test set
import os
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from sqlalchemy import create_engine
data = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
#Databse connection and fetch
engine = create_engine('../input/twitter-airline-sentiment/database.sqlite')
con = engine.connect()
rs = con.execute('select airline_sentiment, text from Tweets')
data_test = pd.DataFrame(rs.fetchall())
data_test.columns = rs.keys()
#Vectorizer
vectorizer = TfidfVectorizer(ngram_range = (1,2))
tfidf_word_freq = vectorizer.fit(data.text)
input_data = vectorizer.transform(data_test.text)
model = keras.models.load_model('Sentiment_analysis_module_2.HD5')
preds = model.predict(input_data)
def get_index(row):
    x = max(row)
    if x == row[0]:
        return 0;
    if x == row[1]:
        return 1;
    return 2

result_set = ['negative', 'neutral', 'positive']
count = 0
for row in range(len(preds)):
    if (result_set[get_index(preds[row])] == data_test.airline_sentiment[row]):
        count+=1
print('Accuracy on test set using NN: ', count/len(preds))