# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import numpy # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import string

from textblob import TextBlob

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize



from nltk import FreqDist

from nltk.stem import SnowballStemmer,WordNetLemmatizer

stemmer=SnowballStemmer('english')

lemma=WordNetLemmatizer()

from string import punctuation

import re





import gc

from keras.preprocessing import sequence,text

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.tsv",sep="\t")

test = pd.read_csv("../input/test.tsv",sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
train.head()
test.head()
def clean_review(reviews):

    reviews_clean = []

    for i in range(0, len(reviews)):

        review = str(reviews[i])

        review = re.sub('[^a-zA-Z]', ' ', review) # regular expression

        review = [

            lemma.lemmatize(w) 

            for w in word_tokenize(str(review).lower())

        ]

        review=' '.join(review)

        reviews_clean.append(review)

        

    return reviews_clean
train['CleanedPhrase'] = clean_review(train.Phrase.values)

train.head()
test['CleanedPhrase'] = clean_review(test.Phrase.values)

test.head()
train['WordCount'] = train['CleanedPhrase'].apply(lambda x: len(TextBlob(x).words))

test['WordCount'] = test['CleanedPhrase'].apply(lambda x: len(TextBlob(x).words))
train = train[train['WordCount'] >= 1]

train = train.reset_index(drop = True)
train.head()
test.head()
sentiment_overview = train.groupby('Sentiment')['WordCount'].describe().reset_index()

sentiment_overview
import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

import matplotlib as plt

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline

import cufflinks as cf

cf.go_offline()
min_length = go.Bar(

    x = sentiment_overview['Sentiment'],

    y = sentiment_overview['min'],

    name = 'Min zinslength'

)



average_length = go.Bar(

    x = sentiment_overview['Sentiment'],

    y = sentiment_overview['mean'],

    name = 'Average Sentence length'

)



max_length = go.Bar(

    x = sentiment_overview['Sentiment'],

    y = sentiment_overview['max'],

    name = 'Max Sentence length'

)



data = [min_length, average_length, max_length]

layout = go.Layout(

    barmode = 'group',

    title = 'Lengte van de zin per sentiment'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')
from sklearn.cluster import KMeans

import numpy as np



x = np.array(train['WordCount'])

km = KMeans(n_clusters = 4)

km.fit(x.reshape(-1,1))  

train['cluster'] = list(km.labels_)
y = np.array(test['WordCount'])

km = KMeans(n_clusters = 4)

km.fit(y.reshape(-1,1))  

test['cluster'] = list(km.labels_)
cluster = train.groupby(['Sentiment','cluster'])['WordCount'].describe().reset_index()

cluster
train.groupby(['Sentiment','cluster'])['WordCount'].count().unstack().plot(kind='bar', stacked=False)

train.groupby(['Sentiment','cluster'])['WordCount'].mean().unstack().plot(kind='bar', stacked=False)

train.groupby(['Sentiment','cluster'])['WordCount'].min().unstack().plot(kind='bar', stacked=False)

train.groupby(['Sentiment','cluster'])['WordCount'].max().unstack().plot(kind='bar', stacked=False)
gc.collect()
train_text = train.filter(['CleanedPhrase','cluster'])

test_text = test.filter(['CleanedPhrase','cluster'])

target = train.Sentiment.values

y = to_categorical(target)



print(train_text.shape,target.shape,y.shape)
X_train_text, X_val_text, y_train, y_val = train_test_split(train_text, y, test_size = 0.2, stratify = y, random_state = 123) # split train + validation



print(X_train_text.shape, y_train.shape)

print(X_val_text.shape, y_val.shape)
all_words = ' '.join(X_train_text.CleanedPhrase.values)

all_words = word_tokenize(all_words)

dist = FreqDist(all_words)

unique_word_count = len(dist)

unique_word_count
review_length = []

for text in X_train_text.CleanedPhrase.values:

    word = word_tokenize(text)

    l = len(word)

    review_length.append(l)

    

max_review_length = np.max(review_length)

max_review_length
max_features = unique_word_count

max_words = max_review_length

batch_size = 128

epochs = 3

num_classes=5
tokenizer = Tokenizer(num_words = max_features)

tokenizer.fit_on_texts(list(X_train_text.CleanedPhrase.values))

X_train = tokenizer.texts_to_sequences(X_train_text.CleanedPhrase.values)

X_val = tokenizer.texts_to_sequences(X_val_text.CleanedPhrase.values)

X_test = tokenizer.texts_to_sequences(test.CleanedPhrase.values)
X_train = sequence.pad_sequences(X_train, maxlen = max_words)

X_val = sequence.pad_sequences(X_val, maxlen = max_words)

X_test = sequence.pad_sequences(X_test, maxlen = max_words)

print(X_train.shape, X_val.shape, X_test.shape)
X_train = numpy.insert(X_train, 48, numpy.array([X_train_text.cluster.values]), axis = 1)

X_val = numpy.insert(X_val, 48, numpy.array([X_val_text.cluster.values]), axis = 1)

X_test = numpy.insert(X_test, 48, numpy.array([test.cluster.values]), axis=1)

print(X_train.shape, X_val.shape, X_test.shape)
gc.collect()
model = Sequential()

model.add(Embedding(max_features, 100, mask_zero = True))

model.add(LSTM(64, dropout = 0.4, recurrent_dropout = 0.4, return_sequences = True))

model.add(LSTM(32, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = False))

model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

model.summary()
%%time

history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, batch_size = batch_size, verbose=1)
prediction = model.predict_classes(X_test, verbose = 1)

sub.Sentiment = prediction

sub.to_csv('sub.csv', index = False)

sub.head()
unique, counts = numpy.unique(prediction, return_counts = True)

dict(zip(unique, counts))