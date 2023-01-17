import pandas as pd
import numpy as np
import nltk

from sklearn.feature_extraction.text import HashingVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

import os
for dirname, _, filenames in os.walk('/kaggle/input/eora-nlp-1/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/eora-nlp-1/reviews_train.csv/reviews_train.csv')
df_test = pd.read_csv('/kaggle/input/eora-nlp-1/reviews_test.csv/reviews_test.csv')
df_train.head(3)
X_train = df_train.summary.replace({np.nan: ''}) + ' ' + df_train.reviewText.replace({np.nan: ''})
X_test = df_test.summary.replace({np.nan: ''}) + ' ' + df_test.reviewText.replace({np.nan: ''})
y_train = df_train.overall > 4
X_train.head(3)
y_train.value_counts()
#https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
import re, string, unicodedata
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

for i, text in enumerate(X_train):
    X_train[i] = ' '.join(normalize(nltk.word_tokenize(text)))

for i, text in enumerate(X_test):
    X_test[i] = ' '.join(normalize(nltk.word_tokenize(text)))
lemmatizer = nltk.wordnet.WordNetLemmatizer()
for i, text in enumerate(X_train):
    X_train[i] = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])

for i, text in enumerate(X_test):
    X_test[i] = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
for i, text in enumerate(X_train):
    X_train[i] = ' '.join([stemmer.stem(w) for w in nltk.word_tokenize(text)])

for i, text in enumerate(X_test):
    X_test[i] = ' '.join([stemmer.stem(w) for w in nltk.word_tokenize(text)])
vectorizer = HashingVectorizer(stop_words=ENGLISH_STOP_WORDS, n_features=10000)

X_train_oh = vectorizer.transform(X_train)
X_test_oh = vectorizer.transform(X_test)
clf = LogisticRegression(solver='sag')
clf.fit(X_train_oh, y_train)
y_pred = clf.predict(X_test_oh).astype(int)
pd.DataFrame(data={'ID': df_test.ID, 'is_perfect': y_pred}).to_csv('submission.csv', index=False)