%matplotlib inline

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

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
train = pd.read_csv('../input/train.csv')

train.head()
train['comment_text'] = train['comment_text'].fillna('nan')
test = pd.read_csv('../input/test.csv')

test.head()
test['comment_text'] = test['comment_text'].fillna('nan')
submission = pd.read_csv('../input/sample_submission.csv')

submission.head()
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:

    print(label, (train[label] == 1.0).sum() / len(train))
train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].corr()
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
train['comment_text_stemmed'] = reduce_texts(stem_word, train['comment_text'])

test['comment_text_stemmed'] = reduce_texts(stem_word, test['comment_text'])

train['comment_text_lemmatized'] = reduce_texts(lemmatize_word, train['comment_text'])

test['comment_text_lemmatized'] = reduce_texts(lemmatize_word, test['comment_text'])
train.head()
test.head()
def metric(y_true, y_pred):

    assert y_true.shape == y_pred.shape

    columns = y_true.shape[1]

    column_losses = []

    for i in range(0, columns):

        column_losses.append(log_loss(y_true[:, i], y_pred[:, i]))

    return np.array(column_losses).mean()
def cv(model, X, y, label2binary, n_splits=3):

    def split(X, y):

        return StratifiedKFold(n_splits=n_splits).split(X, y)

    

    def convert_y(y):

        new_y = np.zeros([len(y)])

        for i, val in enumerate(label2binary):

            idx = (y == val).max(axis=1)

            new_y[idx] = i

        return new_y

    

    X = np.array(X)

    y = np.array(y)

    scores = []

    for train, test in tqdm(split(X, convert_y(y)), total=n_splits):

        fitted_model = model(X[train], y[train])

        scores.append(metric(y[test], fitted_model(X[test])))

    return np.array(scores)
label2binary = np.array([

    [0, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 1],

    [0, 0, 0, 0, 1, 0],

    [0, 0, 0, 0, 1, 1],

    [0, 0, 0, 1, 0, 0],

    [0, 0, 0, 1, 0, 1],

    [0, 0, 0, 1, 1, 0],

    [0, 0, 0, 1, 1, 1],

    [0, 0, 1, 0, 0, 0],

    [0, 0, 1, 0, 0, 1],

    [0, 0, 1, 0, 1, 0],

    [0, 0, 1, 0, 1, 1],

    [0, 0, 1, 1, 0, 0],

    [0, 0, 1, 1, 0, 1],

    [0, 0, 1, 1, 1, 0],

    [0, 0, 1, 1, 1, 1],

    [0, 1, 0, 0, 0, 0],

    [0, 1, 0, 0, 0, 1],

    [0, 1, 0, 0, 1, 0],

    [0, 1, 0, 0, 1, 1],

    [0, 1, 0, 1, 0, 0],

    [0, 1, 0, 1, 0, 1],

    [0, 1, 0, 1, 1, 0],

    [0, 1, 0, 1, 1, 1],

    [0, 1, 1, 0, 0, 0],

    [0, 1, 1, 0, 0, 1],

    [0, 1, 1, 0, 1, 0],

    [0, 1, 1, 0, 1, 1],

    [0, 1, 1, 1, 0, 0],

    [0, 1, 1, 1, 0, 1],

    [0, 1, 1, 1, 1, 0],

    [0, 1, 1, 1, 1, 1],

    [1, 0, 0, 0, 0, 0],

    [1, 0, 0, 0, 0, 1],

    [1, 0, 0, 0, 1, 0],

    [1, 0, 0, 0, 1, 1],

    [1, 0, 0, 1, 0, 0],

    [1, 0, 0, 1, 0, 1],

    [1, 0, 0, 1, 1, 0],

    [1, 0, 0, 1, 1, 1],

    [1, 0, 1, 0, 0, 0],

    [1, 0, 1, 0, 0, 1],

    [1, 0, 1, 0, 1, 0],

    [1, 0, 1, 0, 1, 1],

    [1, 0, 1, 1, 0, 0],

    [1, 0, 1, 1, 0, 1],

    [1, 0, 1, 1, 1, 0],

    [1, 0, 1, 1, 1, 1],

    [1, 1, 0, 0, 0, 0],

    [1, 1, 0, 0, 0, 1],

    [1, 1, 0, 0, 1, 0],

    [1, 1, 0, 0, 1, 1],

    [1, 1, 0, 1, 0, 0],

    [1, 1, 0, 1, 0, 1],

    [1, 1, 0, 1, 1, 0],

    [1, 1, 0, 1, 1, 1],

    [1, 1, 1, 0, 0, 0],

    [1, 1, 1, 0, 0, 1],

    [1, 1, 1, 0, 1, 0],

    [1, 1, 1, 0, 1, 1],

    [1, 1, 1, 1, 0, 0],

    [1, 1, 1, 1, 0, 1],

    [1, 1, 1, 1, 1, 0],

    [1, 1, 1, 1, 1, 1],

])
def dummy_model(X, y):

    def _predict(X):

        return np.ones([X.shape[0], 6]) * 0.5

    

    return _predict



cv(dummy_model,

   train['comment_text'],

   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],

   label2binary)
def regression_baseline(X, y):

    tfidf = TfidfVectorizer()

    X_tfidf = tfidf.fit_transform(X)

    columns = y.shape[1]

    regressions = [

        LogisticRegression().fit(X_tfidf, y[:, i])

        for i in range(columns)

    ]

    

    def _predict(X):

        X_tfidf = tfidf.transform(X)

        predictions = np.zeros([len(X), columns])

        for i, regression in enumerate(regressions):

            regression_prediction = regression.predict_proba(X_tfidf)

            predictions[:, i] = regression_prediction[:, regression.classes_ == 1][:, 0]

        return predictions

    

    return _predict
cv(regression_baseline,

   train['comment_text'],

   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],

   label2binary)
cv(regression_baseline,

   train['comment_text_stemmed'],

   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],

   label2binary)
cv(regression_baseline,

   train['comment_text_lemmatized'],

   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],

   label2binary)
def regression_wordchars(X, y):

    tfidf_word = TfidfVectorizer()

    X_tfidf_word = tfidf_word.fit_transform(X[:, 1])

    tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)

    X_tfidf_char = tfidf_char.fit_transform(X[:, 0])

    X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])

    

    columns = y.shape[1]

    regressions = [

        LogisticRegression().fit(X_tfidf, y[:, i])

        for i in range(columns)

    ]

    

    def _predict(X):

        X_tfidf_word = tfidf_word.transform(X[:, 1])

        X_tfidf_char = tfidf_char.transform(X[:, 0])

        X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])

        predictions = np.zeros([len(X), columns])

        for i, regression in enumerate(regressions):

            regression_prediction = regression.predict_proba(X_tfidf)

            predictions[:, i] = regression_prediction[:, regression.classes_ == 1][:, 0]

        return predictions

    

    return _predict
cv(regression_wordchars,

   train[['comment_text', 'comment_text_stemmed']],

   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],

   label2binary)
%%time

model = regression_wordchars(np.array(train[['comment_text', 'comment_text_stemmed']]),

                             np.array(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]))
%%time

prediction = model(np.array(test[['comment_text', 'comment_text_stemmed']]))
for i, label in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):

    submission[label] = prediction[:, i]

submission.head()
submission.to_csv('output.csv', index=None)