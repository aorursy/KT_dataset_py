import numpy as np

import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



# Scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer



# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM

from keras import utils

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



# nltk

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer



# Word2vec

import gensim



# Utility

import re

import os

from collections import Counter

import logging

import time

import pickle

import itertools

import gc

import json

from keras_preprocessing.text import tokenizer_from_json

from keras.models import model_from_json



pd.set_option('max_colwidth', 500)

pd.set_option('max_columns', 500)

pd.set_option('max_rows', 100)
alexa = pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv' , delimiter = '\t' 

                    ,usecols = ['verified_reviews' , 'feedback'] )
alexa = alexa.rename(columns={'verified_reviews':'reviews', 'feedback':'sentiment'})

display(alexa['sentiment'].value_counts()/alexa.shape[0]*100)

print('Shape of Dataset -> ' , alexa.shape)

display(alexa.sample(6))
twitter = pd.read_csv('../input/twitter-sentiment/Sentiment Analysis Dataset 2.csv', skiprows=[8835,535881] , usecols = ['Sentiment' , 'SentimentText'])

twitter = twitter.rename(columns = {'Sentiment': 'sentiment' , 'SentimentText':'reviews'})

display(twitter['sentiment'].value_counts()/twitter.shape[0]*100)

print('Shape of Dataset -> ' , twitter.shape)

display(twitter.sample(6))
imdb = pd.read_csv('/kaggle/input/imdb-review-dataset/imdb_master.csv', encoding = "ISO-8859-1")

imdb=imdb[imdb['label']!='unsup']

#Preprocessing

imdb=imdb.drop(['Unnamed: 0','type','file'],axis=1)

imdb.label[imdb.label == 'neg'] = 0

imdb.label[imdb.label == 'pos'] = 1

imdb=imdb.rename(columns = {'label': 'sentiment' , 'review':'reviews'})

display(imdb['sentiment'].value_counts()/imdb.shape[0]*100)

print('Shape of Dataset -> ' , imdb.shape)

display(imdb.sample(6))
data = pd.concat([alexa, twitter , imdb], axis= 0)

del alexa , twitter , imdb

gc.collect()
print(data.shape)

display(data.sample(5))

data['sentiment'].value_counts()/data.shape[0]*100
data = data.sample(frac= 0.05 , random_state = 10)
from spacy.lang.en.stop_words import STOP_WORDS

# stop_words = stopwords.words("english")

stop_words = STOP_WORDS

stemmer = SnowballStemmer("english")

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|<.*?>|[^A-Za-z0-9]+"

Emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)



def preprocess(text, stem=False):

    # Remove link,user and special characters

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    text = re.sub(Emoji_pattern, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    return " ".join(tokens)
%%time

data.reviews = data.reviews.apply(lambda x: preprocess(x))
data.head(5)
%%time

documents = [_text.split() for _text in data.reviews] 
W2V_SIZE = 300

W2V_WINDOW = 7

W2V_EPOCH = 32

W2V_MIN_COUNT = 10

w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=8)





w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
%%time

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
w2v_model.most_similar(positive=['awesome'],topn=100)

# [x[0] for x in w2v_model.most_similar("awesome")]
from wordcloud import WordCloud

plt.figure(figsize=(10,5))

wordcloud = WordCloud(background_color="white",

                      stopwords = STOP_WORDS,

                      max_words=45,

                      max_font_size=30,

                      random_state=42

                     ).generate(str([x[0] for x in w2v_model.most_similar("fantastic",topn=100)]))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("SIMILAR WORDS FOR FANTASTIC")

plt.show()
plt.figure(figsize=(10,5))

wordcloud = WordCloud(background_color="white",

                      stopwords = STOP_WORDS,

                      max_words=45,

                      max_font_size=30,

                      random_state=42

                     ).generate(str([x[0] for x in w2v_model.most_similar("poor",topn=100)]))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("SIMILAR WORDS FOR POOR")

plt.show()
tokenizer = Tokenizer()

tokenizer.fit_on_texts(data.reviews)

vocab_size = len(tokenizer.word_index)+1

print('Vocab Size is ',vocab_size)
SEQUENCE_LENGTH = 300

EPOCHS = 8

BATCH_SIZE = 1024
%%time 

x_data = pad_sequences(tokenizer.texts_to_sequences(data.reviews) , maxlen = SEQUENCE_LENGTH)
y_data = data.sentiment

print(x_data.shape)

print(y_data.shape)

y_data = y_data.values.reshape(-1,1)
w2v_model.wv['sample'].shape
embedding_matrix = np.zeros((vocab_size , W2V_SIZE))

for word , i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)
embedding_layer = Embedding( vocab_size , W2V_SIZE , weights = [embedding_matrix] , input_length = SEQUENCE_LENGTH, trainable = False)

model = Sequential()

model.add(embedding_layer)

model.add(LSTM(128 , dropout = 0.2 , recurrent_dropout = 0.2 ,return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(64, dropout = 0.2 , recurrent_dropout = 0.2 ))

model.add(Dropout(0.1))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

# ReduceLRonPlateau is to reduce Learning rate when model stopeed improving

# Early Stopping to stop learning when staturation is reached.
%%time 

history = model.fit(x_data , y_data , batch_size = BATCH_SIZE , epochs = EPOCHS , validation_split = 0.1  , verbose = 1 , callbacks = callbacks)
def predict(text):

    start_at = time.time()

    # Tokenize text

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)

    # Predict

    score = model.predict([x_test])[0]



    return {"score": float(score),

       "elapsed_time": time.time()-start_at}  
print(predict('i am Happy'))

print(predict('i not feeling so great .Little Rest can help but you decide what should i do next '))

print(predict('i am sitting in library for 6 hours . i learned alot but i am tired'))

print(predict('i am tired'))

print(predict('good is not good'))

print(predict('bad is not good'))

print(predict('good is not bad'))

print(predict('how i can end up here'))
model.save_weights('model_weights.h5')

with open('model_architecture.json', 'w') as f:

    f.write(model.to_json())

    

model.save('entire_model.h5')

tokenizer_json = tokenizer.to_json()

with open('tokenizer.json', 'w', encoding='utf-8') as f:

    f.write(json.dumps(tokenizer_json, ensure_ascii=False))