#load libraries for data manipulation and visualization

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

# text/file processing libraries

import string

import re

import sys

from nltk.corpus import stopwords

from itertools import chain

# warnings

import string

import warnings

warnings.filterwarnings('ignore')
# load the train and test data sets

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print('Number of Training Samples = {}'.format(train_df.shape[0]))

print('Number of Test Samples = {}\n'.format(test_df.shape[0]))

print('Training X Shape = {}'.format(train_df.shape))

print('Training y Shape = {}\n'.format(train_df['target'].shape[0]))

print('Test X Shape = {}'.format(test_df.shape))



print('Test y Shape = {}\n'.format(test_df.shape[0]))

print('Index of Train Set:\n', train_df.columns)

print('Index of Test Set:\n', test_df.columns)
# class distribution of train set

pl = sb.countplot(train_df['target'])
# display sample train data

train_df.sample(5)
# sample train disaster tweet

train_df.loc[1241]['text']
# sample train non disaster tweet

train_df.loc[2301]['text']
train_df['text'].sample(20).tolist()
def html_references(tweets):

    texts = tweets

    # remove url - references to websites

    url_remove  = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    texts  = re.sub(url_remove, '', texts)

    # remove common html entity references in utf-8 as '&lt;', '&gt;', '&amp;'

    entities_remove = r'&amp;|&gt;|&lt'

    texts = re.sub(entities_remove, "", texts)

    # split into words by white space

    words = texts.split()

    #convert to lower case

    words = [word.lower() for word in words]

    return " ".join(words)

train_df['text'] = train_df['text'].apply(lambda x : html_references(x))

test_df['text'] = test_df['text'].apply(lambda x : html_references(x))
def decontraction(tweet):

    # specific

    tweet = re.sub(r"won\'t", " will not", tweet)

    tweet = re.sub(r"won\'t've", " will not have", tweet)

    tweet = re.sub(r"can\'t", " can not", tweet)

    tweet = re.sub(r"don\'t", " do not", tweet)

    

    tweet = re.sub(r"can\'t've", " can not have", tweet)

    tweet = re.sub(r"ma\'am", " madam", tweet)

    tweet = re.sub(r"let\'s", " let us", tweet)

    tweet = re.sub(r"ain\'t", " am not", tweet)

    tweet = re.sub(r"shan\'t", " shall not", tweet)

    tweet = re.sub(r"sha\n't", " shall not", tweet)

    tweet = re.sub(r"o\'clock", " of the clock", tweet)

    tweet = re.sub(r"y\'all", " you all", tweet)

    # general

    tweet = re.sub(r"n\'t", " not", tweet)

    tweet = re.sub(r"n\'t've", " not have", tweet)

    tweet = re.sub(r"\'re", " are", tweet)

    tweet = re.sub(r"\'s", " is", tweet)

    tweet = re.sub(r"\'d", " would", tweet)

    tweet = re.sub(r"\'d've", " would have", tweet)

    tweet = re.sub(r"\'ll", " will", tweet)

    tweet = re.sub(r"\'ll've", " will have", tweet)

    tweet = re.sub(r"\'t", " not", tweet)

    tweet = re.sub(r"\'ve", " have", tweet)

    tweet = re.sub(r"\'m", " am", tweet)

    tweet = re.sub(r"\'re", " are", tweet)

    return tweet 

train_df['text'] = train_df['text'].apply(lambda x : decontraction(x))

test_df['text'] = test_df['text'].apply(lambda x : decontraction(x))
# print puntuation characters

string.punctuation
# print printable characters

string.printable
def filter_punctuations_etc(tweets):

    words = tweets.split()

    # prepare regex for char filtering

    re_punc = re.compile( '[%s]' % re.escape(string.punctuation))

    # remove punctuation from each word

    words = [re_punc.sub('', w) for w in words]

    # filter out non-printable characters

    re_print = re.compile( '[^%s]' % re.escape(string.printable))

    words = [re_print.sub(' ', w) for w in words]

    return " ".join(words)

train_df['text'] = train_df['text'].apply(lambda x : filter_punctuations_etc(x))

test_df['text'] = test_df['text'].apply(lambda x : filter_punctuations_etc(x))
def separate_alphanumeric(tweets):

    words = tweets

    # separate alphanumeric

    words = re.findall(r"[^\W\d_]+|\d+", words)

    return " ".join(words)

train_df['text'] = train_df['text'].apply(lambda x : separate_alphanumeric(x))

test_df['text'] = test_df['text'].apply(lambda x : separate_alphanumeric(x))
def cont_rep_char(text):

    tchr = text.group(0) 

    

    if len(tchr) > 1:

        return tchr[0:2] # take max of 2 consecutive letters

def unique_char(rep, tweets):

    substitute = re.sub(r'(\w)\1+', rep, tweets)

    return substitute

train_df['text'] = (train_df['text'].astype('str').apply(lambda x : unique_char(cont_rep_char, x)))

test_df['text'] = (test_df['text'].astype('str').apply(lambda x : unique_char(cont_rep_char, x)))
!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)

#train_df['text'] = train_df['text'].apply(lambda x : correct_spellings(x))

#test_df['text'] = test_df['text'].apply(lambda x : correct_spellings(x))
!pip install wordninja
import wordninja # !pip install wordninja

def split_attached_words(tweet):

    words = wordninja.split(tweet)

    return" ".join(words)

train_df['text'] = train_df['text'].apply(lambda x : split_attached_words(x))

test_df['text'] = test_df['text'].apply(lambda x : split_attached_words(x))
def stopwords_shortwords(tweet):

    # filter out stop words

    words = tweet.split()

    stop_words = set(stopwords.words( 'english' ))

    words = [w for w in words if not w in stop_words]

    # filter out short tokens

    for word in words:

        if word.isalpha():

            words = [word for word in words if len(word) > 1 ]

        else:

            words = [word for word in words]

    return" ".join(words)

train_df['text'] = train_df['text'].apply(lambda x : stopwords_shortwords(x))

test_df['text'] = test_df['text'].apply(lambda x : stopwords_shortwords(x))
from sklearn.model_selection import train_test_split

# split train set into train/validate 

train_df2, validate_df = train_test_split(train_df, test_size=0.075, random_state=0)

train_df2 = train_df2.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
# train and test texts

all_df=pd.concat([train_df,test_df])

X_all = all_df['text']

# training set

X_train = train_df2['text']

y_train = train_df2['target'].astype(int)

# validation set

X_validate= validate_df['text']

y_validate = validate_df['target'].astype(int)

# test set

X_test = test_df['text']
from tensorflow.keras.preprocessing.text import Tokenizer

# fit a tokenizer

def create_tokenizer(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer
# create the tokenizer

tokenizer = create_tokenizer(X_all)

# encode data

Xtrain = tokenizer.texts_to_matrix(X_train, mode = 'tfidf' )

Xvalidate = tokenizer.texts_to_matrix(X_validate, mode = 'tfidf' )

Xtest = tokenizer.texts_to_matrix(X_test, mode = 'tfidf' )

# vocabulary

n_words = Xtest.shape[1]

print('There are ' + str(n_words) + ' words in the vocabulary')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras import layers

import  tensorflow.keras.optimizers as optimizers

# define a basic model

def define_model(n_words):

    # define network

    model = Sequential()

    model.add(Dense(32, input_shape=(n_words,), activation= 'relu' ))

    model.add(Dense(64, activation= 'relu' ))

    model.add(Dense(1, activation= 'sigmoid' ))

    # compile network

    model.compile(loss= 'binary_crossentropy' , optimizer=optimizers.Adam(lr=.0001) , metrics=[ 'accuracy' ])

    # summarize defined model

    model.summary()

    return model
# optimizing model performance

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [

    EarlyStopping(patience=5, verbose=1),

    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=0.00001, verbose=1),

    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)

]
# define the model

model = define_model(n_words)

# fit network

model.fit(Xtrain, y_train, epochs=10, callbacks=callbacks, validation_data=(Xvalidate,y_validate))
sample_submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

y_pre=model.predict(Xtest)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_submission['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission2.csv',index=False)