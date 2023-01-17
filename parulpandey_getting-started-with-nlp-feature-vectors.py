# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import string

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
sentences = ['The weather is sunny', 'The weather is partly sunny and partly cloudy.']
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences)
vectorizer.vocabulary_
# Converting all the sentences to arrays
vectorizer.transform(sentences).toarray()
stopwords = stopwords.words('english')

count_vectorizer = CountVectorizer(stop_words = stopwords)
count_vectorizer.fit(train['text'])

train_vectors = count_vectorizer.transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])

train_vectors.shape
count_vectorizer = CountVectorizer(stop_words = stopwords, min_df=2 ,max_df=0.8)
count_vectorizer.fit(train['text'])

train_vectors = count_vectorizer.transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])
# Creating a custom preprocessor that lowercases, removes special characters, removes hyperlinks and punctuation

def custom_preprocessor(text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text

    
count_vectorizer = CountVectorizer(list(train['text']),preprocessor=custom_preprocessor)

train_vectors = count_vectorizer.fit_transform(list(train['text']))
test_vectors = count_vectorizer.transform(list(test['text']))
 
# World level unigrams and bigrams

count_vectorizer = CountVectorizer(list(train['text']),preprocessor=custom_preprocessor,ngram_range=(1,2))

train_vectors = count_vectorizer.fit_transform(list(train['text']))
test_vectors = count_vectorizer.transform(list(test['text']))

list(count_vectorizer.vocabulary_)[:10]
# character level bigrams


count_vectorizer = CountVectorizer(list(train['text']),preprocessor=custom_preprocessor,ngram_range=(2,2),
                                  analyzer='char_wb')

train_vectors = count_vectorizer.fit_transform(list(train['text']))
test_vectors = count_vectorizer.transform(list(test['text']))

print(list(count_vectorizer.vocabulary_)[:20])

count_vectorizer = CountVectorizer(token_pattern=r'\w{1,}',
                   ngram_range=(1, 2), stop_words = stopwords,preprocessor=custom_preprocessor)
count_vectorizer .fit(train['text'])

train_vectors = count_vectorizer.transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")
scores
# Fitting a simple Logistic Regression on Counts
clf.fit(train_vectors, train["target"])
# Submission
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv("submission.csv", index=False)
# word level
tfidf = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}',max_features=5000)
train_tfidf = tfidf.fit_transform(train['text'])
test_tfidf = tfidf.transform(test["text"])
#ngram level
tfidf = TfidfVectorizer(analyzer='word',ngram_range=(2,3),token_pattern=r'\w{1,}',max_features=5000)
train_tfidf = tfidf.fit_transform(train['text'])
test_tfidf = tfidf.transform(test["text"])
# characters level
tfidf = TfidfVectorizer(analyzer='char',ngram_range=(2,3),token_pattern=r'\w{1,}',max_features=5000)
train_tfidf = tfidf.fit_transform(train['text'])
test_tfidf = tfidf.transform(test["text"])

tfidf_vectorizer = TfidfVectorizer( min_df=3,  max_features=None,analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = stopwords)

train_tfidf = tfidf.fit_transform(train['text'])
test_tfidf = tfidf.transform(test["text"])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(clf, train_tfidf, train["target"], cv=5, scoring="f1")
scores
# Fitting a simple Logistic Regression on TFIDF
clf.fit(train_tfidf, train["target"])
hash_vectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False)
hash_vectorizer.fit(train['text'])


train_vectors = hash_vectorizer.transform(train['text'])
test_vectors = hash_vectorizer.transform(test['text'])
print(train_vectors[0])
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")
scores
# Fitting a simple Logistic Regression on TFIDF
clf.fit(train_vectors, train["target"])