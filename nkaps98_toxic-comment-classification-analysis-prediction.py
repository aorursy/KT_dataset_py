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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

#from textblob import TextBlob

#from spellchecker import SpellChecker

#from autocorrect import spell

#from gingerit.gingerit import GingerIt

#from symspellpy.symspellpy import SymSpell, Verbosity

#from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec

#from glove import Glove, Corpus

import tensorflow

from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding, Flatten, SimpleRNN, RNN,GRU, SpatialDropout1D, Dropout

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import text_to_word_sequence

dataset = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

X = dataset.iloc[:,1].values

y = dataset.iloc[:,2:].values



dataset_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

X_test = dataset_test.iloc[:,1].values

X_test = X_test.reshape(153164,1)



test_labels = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')

Y_test = test_labels.iloc[:,1:].values



merged = pd.merge(dataset_test, test_labels, how="left", on="id")
dataset.shape
merged.head()

Y_test
merged['sum'] = merged['toxic'] + merged['severe_toxic'] + merged['obscene'] + merged['threat'] + merged['insult'] + merged['identity_hate']
merged.drop('id',axis=1, inplace=True)

merged.head()
merge = merged[merged['sum'] != -6]
merge.shape
Y_test = merge.iloc[:,1:7].values

X_test = merge.iloc[:,0].values
X_test.shape
tokens = []

tokens = [word_tokenize(str(sentence)) for sentence in X]

rm = []

for w in tokens:

    sm = re.sub('[^A-Za-z]',' ', str(w))

    x = re.split("\s", sm)

    rm.append(x)



#Removing whitespaces

for sent in rm:

    while '' in sent:

        sent.remove('')



# Lowercasing

low = []

for i in rm:

    i = [x.lower() for x in i]

    low.append(i)

lemma = []

wnl = WordNetLemmatizer()

for doc in low:

    tokens = [wnl.lemmatize(w) for w in doc]

    lemma.append(tokens)



# Removing Stopwords

filter_words = []

Stopwords = set(stopwords.words('english'))



#ab = spell('nd')

for sent in lemma:

    tokens = [w for w in sent if w not in Stopwords]

    filter_words.append(tokens)



space = ' ' 

sentences = []

for sentence in filter_words:

    sentences.append(space.join(sentence))

filtered_words = []

for sent in filter_words:

    token = [word for word in sent if len(word)>2]

    filtered_words.append(token)
model_cbow = Word2Vec(filtered_words)

word_vectors = model_cbow.wv

vocabulary = word_vectors.vocab.items()

model_cbow.most_similar('mother')
len(word_vectors.vocab)
keys = list(word_vectors.vocab.keys())

unk = 0

total = 0



embedding_matrix = word_vectors.vectors

## Word with their index values

word2id = {k:v.index for k,v in word_vectors.vocab.items()}



## Unknown values

UNK_INDEX = 0

UNK_TOKEN = 'UNK'

unk_vector = embedding_matrix.mean(0)



## Inserting row for unknown words 

embedding_matrix = np.insert(embedding_matrix, [UNK_INDEX], [unk_vector], axis=0)

word2id = {word:(index+1) if index >= UNK_INDEX else index for word, index in 

           word2id.items()}

word2id[UNK_TOKEN] = UNK_INDEX



## Replacing words in x_train with their respective indices and replacing each unknown 

## word with index 0

L = []

for sent in filter_words:

    Z = []

    for word in sent:

        if word in word2id:

            Z.append(word2id.get(word))

        else:

            Z.append(UNK_INDEX)

            unk+=1

    L.append(Z)

X_train = pad_sequences(L, maxlen=100, padding='post',

                        dtype='float')

## Implementing RNN using GRU/LSTM

vocab_len = len(embedding_matrix)

model = Sequential()

model.add(Embedding(vocab_len, 100, input_length = 100,weights=[embedding_matrix]))

model.add(GRU(units=100, activation='tanh'))

#model.add(LSTM(units=120, activation='tanh'))

model.add(Dropout(0.2))

model.add(Dense(50,activation='tanh'))

model.add(Dense(6,activation='softmax'))

model.compile(optimizer='adam',loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()

model.fit(X_train,y,batch_size=1000,epochs=10)

import pandas as pd

x_test = []

for sentence in X_test:

    x_test.append(text_to_word_sequence(str(sentence),filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '))



filter_test = []

for sent in x_test:

    tokens = [w for w in sent if w not in Stopwords]

    filter_test.append(tokens)



## Converting the text into sequences using ids

L = []

for sent in x_test:

    Z = []

    for word in sent:

        if word in word2id and len(word)>2:

            Z.append(word2id.get(word))

        else:

            Z.append(UNK_INDEX)

            unk+=1

    L.append(Z)



X_test = pad_sequences(L, maxlen=100, padding= 'post',dtype='float')

y_pred = model.predict(X_test)

model.evaluate(X_test,Y_test)