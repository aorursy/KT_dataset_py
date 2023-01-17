import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import PorterStemmer

import nltk

from nltk import word_tokenize, ngrams

from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
np.random.seed(25)

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.layers.normalization import BatchNormalization

from keras import backend as K

from keras.layers import Convolution1D, GlobalMaxPooling1D

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.layers.merge import concatenate

from keras.layers.core import Dense, Activation, Dropout

import codecs
train = pd.read_csv("../input/hotel-review/train.csv")

test = pd.read_csv("../input/hotel-review/test.csv")
# Target Mapping

mapping_target = {'happy':0, 'not happy':1}

train = train.replace({'Is_Response':mapping_target})



# Browser Mapping

mapping_browser = {'Firefox':0, 'Mozilla':0, 'Mozilla Firefox':0,

                  'Edge': 1, 'Internet Explorer': 1 , 'InternetExplorer': 1, 'IE':1,

                   'Google Chrome':2, 'Chrome':2,

                   'Safari': 3, 'Opera': 4

                  }

train = train.replace({'Browser_Used':mapping_browser})

test = test.replace({'Browser_Used':mapping_browser})

# Device mapping

mapping_device = {'Desktop':0, 'Mobile':1, 'Tablet':2}

train = train.replace({'Device_Used':mapping_device})

test = test.replace({'Device_Used':mapping_device})
GLOVE_DIR = '../input/glove-global-vectors-for-word-representation/'

MAX_SEQUENCE_LENGTH = 300

MAX_NB_WORDS = 10000

EMBEDDING_DIM = 32

VALIDATION_SPLIT = 0.3
test_id = test['User_ID']

target = train['Is_Response']
# function to clean data

import string

import itertools 

import re

from nltk.stem import WordNetLemmatizer

from string import punctuation



stops = ['also','on','the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',

              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',

              'Is','If','While','This','january','february','march','april','may','june','july','august','september','october',

        'november','december','monday','tuesday','wednesday','thursday','friday','saturday','sunday','india','tripadviser','usa',

        'hundred','thousand','today','tomorrow','yesterday','etc','delhi','mumbai','chennai','kolkata','room','hotel','even',

        'front desk','new york','san francisco','however','time square','canada','review','us','uk','china','staff','found'

        ,'one','area','although','walking', 'distance','though','th','floor','really','got','people','lobby','location'] 

# punct = list(string.punctuation)

# punct.append("''")

# punct.append(":")

# punct.append("...")

# punct.append("@")

# punct.append('""')

def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):

    txt = str(text)

    

    # Replace apostrophes with standard lexicons

    txt = txt.replace("isn't", "is not")

    txt = txt.replace("aren't", "are not")

    txt = txt.replace("ain't", "am not")

    txt = txt.replace("won't", "will not")

    txt = txt.replace("didn't", "did not")

    txt = txt.replace("shan't", "shall not")

    txt = txt.replace("haven't", "have not")

    txt = txt.replace("hadn't", "had not")

    txt = txt.replace("hasn't", "has not")

    txt = txt.replace("don't", "do not")

    txt = txt.replace("wasn't", "was not")

    txt = txt.replace("weren't", "were not")

    txt = txt.replace("doesn't", "does not")

    txt = txt.replace("'s", " is")

    txt = txt.replace("'re", " are")

    txt = txt.replace("'m", " am")

    txt = txt.replace("'d", " would")

    txt = txt.replace("'ll", " will")

    txt = txt.replace("--th", " ")

    

    # More cleaning

    txt = re.sub(r"alot", "a lot", txt)

    txt = re.sub(r"what's", "", txt)

    txt = re.sub(r"What's", "", txt)

    txt = re.sub(r"\'s", " ", txt)

    txt = txt.replace("pic", "picture")

    txt = re.sub(r"\'ve", " have ", txt)

    txt = re.sub(r"can't", "cannot ", txt)

    txt = re.sub(r"n't", " not ", txt)

    txt = re.sub(r"I'm", "I am", txt)

    txt = re.sub(r" m ", " am ", txt)

    txt = re.sub(r"\'re", " are ", txt)

    txt = re.sub(r"\'d", " would ", txt)

    txt = re.sub(r"\'ll", " will ", txt)

    txt = re.sub(r"60k", " 60000 ", txt)

    txt = re.sub(r" e g ", " eg ", txt)

    txt = re.sub(r" b g ", " bg ", txt)

    txt = re.sub(r"\0s", "0", txt)

    txt = re.sub(r" 9 11 ", "911", txt)

    txt = re.sub(r"e-mail", "email", txt)

    txt = re.sub(r"\s{2,}", " ", txt)

    txt = re.sub(r"quikly", "quickly", txt)

    txt = re.sub(r"imrovement", "improvement", txt)

    txt = re.sub(r"intially", "initially", txt)

    txt = re.sub(r"quora", "Quora", txt)

    txt = re.sub(r" dms ", "direct messages ", txt)  

    txt = re.sub(r"demonitization", "demonetization", txt) 

    txt = re.sub(r"actived", "active", txt)

    txt = re.sub(r"kms", " kilometers ", txt)

    txt = re.sub(r"KMs", " kilometers ", txt)

    txt = re.sub(r" cs ", " computer science ", txt) 

    txt = re.sub(r" upvotes ", " up votes ", txt)

    txt = re.sub(r" iPhone ", " phone ", txt)

    txt = re.sub(r"\0rs ", " rs ", txt) 

    txt = re.sub(r"calender", "calendar", txt)

    txt = re.sub(r"ios", "operating system", txt)

    txt = re.sub(r"gps", "GPS", txt)

    txt = re.sub(r"gst", "GST", txt)

    txt = re.sub(r"programing", "programming", txt)

    txt = re.sub(r"bestfriend", "best friend", txt)

    txt = re.sub(r"dna", "DNA", txt)

    txt = re.sub(r"III", "3", txt) 

    txt = re.sub(r"the US", "America", txt)

    txt = re.sub(r"Astrology", "astrology", txt)

    txt = re.sub(r"Method", "method", txt)

    txt = re.sub(r"Find", "find", txt) 

    txt = re.sub(r"banglore", "Banglore", txt)

    txt = re.sub(r" J K ", " JK ", txt)

    txt = re.sub(r"comfy", "comfortable", txt)

    txt = re.sub(r"colour", "color", txt)

    txt = re.sub(r"travellers", "travelers", txt)



    # Emoji replacement

#     txt = re.sub(r':\)',r' Happy ',txt)

#     txt = re.sub(r':D',r' Happy ',txt)

#     txt = re.sub(r':P',r' Happy ',txt)

#     txt = re.sub(r':\(',r' Sad ',txt)

    

    # Remove urls and emails

    txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)

    txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)

    

    # Remove punctuation from text

    txt = ''.join([c for c in text if c not in punctuation])

#     txt = txt.replace(".", " ")

#     txt = txt.replace(":", " ")

#     txt = txt.replace("!", " ")

#     txt = txt.replace("&", " ")

#     txt = txt.replace("#", " ")

    

    # Remove all symbols

    txt = re.sub(r'[^A-Za-z0-9\s]',r' ',txt)

    txt = re.sub(r'\n',r' ',txt)

    

    txt = re.sub(r'[0-9]',r' ',txt)

    

    # Replace words like sooooooo with so

    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))

    

    # Split attached words

    #txt = " ".join(re.findall('[A-Z][^A-Z]*', txt))   

    

    if lowercase:

        txt = " ".join([w.lower() for w in txt.split()])

        

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in stops])

    if stemming:

        st = PorterStemmer()

#         print (len(txt.split()))

#         print (txt)

        txt = " ".join([st.stem(w) for w in txt.split()])

    

    if lemmatization:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])



    return txt
# clean description

train['Description'] = train['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = True))

test['Description'] = test['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = True))
# print('Indexing word vectors.')

# embeddings_index = {}

# f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding='utf-8')

# for line in f:

#     values = line.split(' ')

#     word = values[0]

#     coefs = np.asarray(values[1:], dtype='float32')

#     embeddings_index[word] = coefs

# f.close()

# print('Found %s word vectors.' % len(embeddings_index))
print('Processing text dataset')

texts_1 = []

for text in train['Description']:

    texts_1.append(text)



labels = train['Is_Response']  # list of label ids



print('Found %s texts.' % len(texts_1))

test_texts_1 = []

for text in test['Description']:

    test_texts_1.append(text)

print('Found %s texts.' % len(test_texts_1))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts_1 + test_texts_1)

sequences_1 = tokenizer.texts_to_sequences(texts_1)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)



data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.array(labels)

print('Shape of data tensor:', data_1.shape)

print('Shape of label tensor:', labels.shape)



test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)

#test_labels = np.array(test_labels)

del test_sequences_1

del sequences_1

import gc

gc.collect()
# print('Preparing embedding matrix.')

# # prepare embedding matrix

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1



# embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

# for word, i in word_index.items():

#     if i >= nb_words:

#         continue

#     embedding_vector = embeddings_index.get(word)

#     if embedding_vector is not None:

#         # words not found in embedding index will be all-zeros.

#         embedding_matrix[i] = embedding_vector

# print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
# embedding_layer = Embedding(nb_words,

#                             EMBEDDING_DIM,

#                             weights=[embedding_matrix],

#                             input_length=MAX_SEQUENCE_LENGTH,

#                             trainable=False)
from keras.layers.recurrent import LSTM, GRU

model = Sequential()

model.add(Embedding(nb_words,32,input_length=MAX_SEQUENCE_LENGTH))

# model.add(Flatten())

# model.add(Dense(800, activation='relu'))

# model.add(Dropout(0.2))

# model.add(Conv1D(64,

#                  5,

#                  padding='valid',

#                  activation='relu'))

# model.add(Dropout(0.2))

# model.add(MaxPooling1D())

# model.add(Flatten())

# model.add(Dense(400, activation='relu'))

# model.add(Dropout(0.7))



model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dropout(0.2))



model.add(Dense(2, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

# early_stopping =EarlyStopping(monitor='val_loss', patience=3)

# class_weight= {0: 1.309028344, 1: 0.472001959}

model.fit(data_1, to_categorical(labels), validation_split=0.1, nb_epoch=2, batch_size=64)
pred = model.predict(test_data_1)

pred.shape
preds = []



for i in pred:

    if i[0] >= i[1]:

        preds.append('happy')

    else:

        preds.append('not_happy')
result = pd.DataFrame()

result['User_ID'] = test_id

result['Is_Response'] = preds

mapping = {0:'happy', 1:'not_happy'}

result = result.replace({'Is_Response':mapping})



result.to_csv("nn_predicted_result_1.csv", index=False)