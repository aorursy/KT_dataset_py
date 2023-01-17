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
from keras.layers import Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D,GlobalMaxPool1D
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D, concatenate,Concatenate
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Activation, Dropout
import codecs
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import nltk
from nltk import word_tokenize, ngrams
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud,STOPWORDS
import xgboost as xgb
np.random.seed(25)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.shape
train['comment_text'][0]
train['comment_text'][1]
train.isnull().sum(axis=0)
types = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

train[types].describe()
count_list = []
for i in types:
    count_list.append(train[i].sum())
    
sns.barplot(x=types, y=count_list)
types = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

sampled_train1 = train[(train['toxic'] ==0) & (train['severe_toxic'] ==0) & (train['obscene'] ==0)
                       & (train['threat'] ==0) & (train['insult'] ==0) & (train['identity_hate'] ==0)]

sampled_train2 = train[(train['toxic'] !=0) | (train['severe_toxic'] !=0) | (train['obscene'] !=0) 
                | (train['threat'] !=0) | (train['insult'] !=0) | (train['identity_hate'] !=0)]
sampled_train2.head()
sampled_train = sampled_train2.append(sampled_train1)
sampled_train = sampled_train.sample(frac=1)
sampled_train.head()
sampled_train.shape
# function to clean data
import string
import itertools 
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

stop_words = set(stopwords.words('english'))

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
    
    
    # Remove urls and emails
    txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)
    
    # Replace words like sooooooo with so
    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
    
    # Remove punctuation from text
    txt = ''.join([c for c in text if c not in punctuation])
    
    # Remove all symbols
    txt = re.sub(r'[^A-Za-z\s]',r' ',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stop_words])
        
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])
    
    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

    return txt
# clean comments
sampled_train['comment_text'] = sampled_train['comment_text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=False, lemmatization = False))
test['comment_text'] = test['comment_text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=False, lemmatization = False))
MAX_SEQUENCE_LENGTH = 400
MAX_NB_WORDS = 50000 #200000
tokenizer = Tokenizer(lower=False, filters='',num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(sampled_train['comment_text'])

sequences = tokenizer.texts_to_sequences(sampled_train['comment_text'])
test_sequences = tokenizer.texts_to_sequences(test['comment_text'])

train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of train data tensor:', train_data.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

nb_words = (np.max(train_data) + 1)
from keras.layers.recurrent import LSTM, GRU
model = Sequential()
model.add(Embedding(nb_words,50,input_length=MAX_SEQUENCE_LENGTH))
# model.add(SpatialDropout1D(0.2))
# model.add(Bidirectional(GRU(20, return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='sigmoid'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
labels = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
y = sampled_train[labels].values
model.fit(train_data, y, validation_split=0.2, nb_epoch=5, batch_size=32)
pred = model.predict(test_data)
pred[:10]
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[labels] = pred

sample_submission.to_csv("result.csv", index=False)
sample_submission.head()
