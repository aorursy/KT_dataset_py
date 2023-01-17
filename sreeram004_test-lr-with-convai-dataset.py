# loading libraries

import pandas as pd, numpy as np
# fixing seed.!!

seed = 7

np.random.seed(seed)
# output of the kernal https://www.kaggle.com/eoveson/convai-datasets-baseline-models with some tunings

test_new = pd.read_csv('../input/convai-datasets-baseline-models/test_with_convai.csv')

train_new = pd.read_csv('../input/convai-datasets-baseline-models/train_with_convai.csv')
# features we are interesed on

feats_to_concat = ['comment_text', 'toxic_level', 'attack', 'aggression']
# combining test and train

alldata = pd.concat([train_new[feats_to_concat], test_new[feats_to_concat]], axis=0)

alldata.comment_text.fillna('unknown', inplace=True)
# loading libraries

import nltk

nltk.download('wordnet')

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

import re
# define function for cleaning..!!



def cleanData(text, stemming = False, lemmatize=False):

    

    text = text.lower().split()

    text = " ".join(text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    

   

    if stemming:

        st = PorterStemmer()

        txt = " ".join([st.stem(w) for w in text.split()])

        

    if lemmatize:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w) for w in text.split()])





    return text
# cleaning data - stemm and lemm are done later

alldata['comment_text'] = alldata['comment_text'].map(lambda x: cleanData(x,  stemming = False, lemmatize=False))
# again libraries.!!

from matplotlib import pyplot as plt

from nltk.tokenize import wordpunct_tokenize

from nltk.stem.snowball import EnglishStemmer

from nltk.stem import WordNetLemmatizer

from functools import lru_cache

from tqdm import tqdm as tqdm

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from scipy import sparse
# set stopwords

from nltk.corpus import stopwords



eng_stopwords = set(stopwords.words("english"))
# stemming and lemmatizing

# adapted from the kernal 

stemmer = EnglishStemmer()



@lru_cache(30000)

def stem_word(text):

    return stemmer.stem(text)





lemmatizer = WordNetLemmatizer()



@lru_cache(30000)

def lemmatize_word(text):

    return lemmatizer.lemmatize(text)





def reduce_text(conversion, text):

    return " ".join(map(conversion, wordpunct_tokenize(text.lower())))





def reduce_texts(conversion, texts):

    return [reduce_text(conversion, str(text))

            for text in tqdm(texts)]
# lemmatizing and stemming

alldata['comment_text'] = reduce_texts(stem_word, alldata['comment_text'])

alldata['comment_text'] = reduce_texts(lemmatize_word, alldata['comment_text'])
# making placeholder for prediction

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



only_col = ['toxic']



preds = np.zeros((test_new.shape[0], len(col)))
# TfidfVectorizer for words and chars

vect_words = TfidfVectorizer(max_features=40000, analyzer='word', ngram_range=(1, 1))

vect_chars = TfidfVectorizer(max_features=10000, analyzer='char', ngram_range=(1, 3))
# Creating features

all_words = vect_words.fit_transform(alldata.comment_text)

all_chars = vect_chars.fit_transform(alldata.comment_text)
# splitting to train and test

train_words = all_words[:len(train_new)]

test_words = all_words[len(train_new):]



train_chars = all_chars[:len(train_new)]

test_chars = all_chars[len(train_new):]
# needed feats.!!

feats = ['toxic_level', 'attack']
# make sparse matrix with needed data for train and test

train_feats = sparse.hstack([train_words, train_chars, alldata[feats][:len(train_new)]])

test_feats = sparse.hstack([test_words, test_chars, alldata[feats][len(train_new):]])
# libraries.!

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
# fit a LogisticRegression model on full train data and make prediction

for i, j in enumerate(col):

    print('===Fit '+j)

    

    model = LogisticRegression(C=4.0, solver='sag')

    print('Fitting model')

    model.fit(train_feats, train_new[j])

      

    print('Predicting on test')

    preds[:,i] = model.predict_proba(test_feats)[:,1]
# make submission..!!

subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')



submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)

submission.to_csv('feat_lr_2cols.csv', index=False) # 0.045 in the LB