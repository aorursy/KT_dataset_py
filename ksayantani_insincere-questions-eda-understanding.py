# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import re
import time
from string import punctuation

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import namedtuple, defaultdict

from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc

from sklearn.linear_model import LogisticRegression
import gc

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from pprint import pprint
def get_stats(original_function):
    def wraps(*args, **kwargs):
        df = original_function(*args, **kwargs)
        print("nrows : %d" % df.shape[0])
        print("ncolumns : %d" % df.shape[1])
        return df
    return wraps
        
def log_time(original_function):
    def wraps(*args, **kwargs):
        begin = time.time()
        results = original_function(*args, **kwargs)
        print("Elapsed time %fs" % (begin - time.time()))
        return results
    return wraps

@get_stats
def read_csv(path):
    return pd.read_csv(path)
def sort_vocab(func):
    def wraps(*args, **kwargs):
        vectorizer, X = func(*args, **kwargs)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        multiplier = -1
        mask_inds = (multiplier * tfs).argsort()
        
        terms = list(vectorizer.vocabulary_.keys())
        indices = list(vectorizer.vocabulary_.values())
        labels = list()
        for i, index in enumerate(mask_inds):
            labels.append(terms[indices.index(index)])
            
        return labels, list(tfs[mask_inds])
    return wraps

def fit_transform(original_function):
    def wraps(*args, **kwargs):
        corpus = original_function(*args, **kwargs)
        X_train = corpus.apply(lambda doc: " ".join(doc))
        vect = CountVectorizer(lowercase=False).fit(X_train)
        X = vect.transform(X_train)
        return vect, X
    return wraps

def limit_features(func):
    def wraps(*args, **kwargs):
        matches = func(*args, **kwargs)
        pos = np.where(matches.apply(len) > 0)[0]
        return matches[pos]
    return wraps

@sort_vocab
@fit_transform
@limit_features
def search_pattern(raw_documents, use_vect=True, token_pattern=None):
    if use_vect:
        pat = re.compile(token_pattern)
        return raw_documents.apply(lambda doc: pat.findall(doc))
    
    return raw_documents.apply(lambda doc: " ".join(doc))

def plot_barh(response, title='', color='m', N=5, xlim=None, figsize=(9, 8)):
    '''Plots a horizontal bar chart'''
    labels, values = response
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(N)
    rects = ax.barh(y_pos, values[:N], color=color, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[:N], fontdict={'size': 15, 'color': 'b'})
    ax.set_title(title, fontdict={'size': 20})
    if xlim is not None:
        ax.set_xlim(xlim)
        
    for rect in rects:
        w = rect.get_width()
        xloc = w - 10
        yloc = rect.get_y() + (rect.get_height()/2.0)
        ax.text(xloc, yloc, "%.5f" % w if w < 1 else w, horizontalalignment='center', verticalalignment='center', color='white', weight='bold', fontdict={'size': 14}, clip_on=True)
        
    plt.show()
CloudParams = namedtuple('CloudParams', ['title', 'size', 'color'])

def prepare_cloud_data(corpus):
    return " ".join(corpus)

def prepare_cloud_data_from_tokens(corpus):
    wordcloud = ' '
    for item in np.hstack(corpus):
        wordcloud = wordcloud + ' ' + item
    return wordcloud

def plot_wordcloud(text, stops, params, figsize=(8, 6)):
    stopwords = stop_words.ENGLISH_STOP_WORDS
    if stops is not None:
        stopwords = stopwords.union(stops)
        
    wordcloud = WordCloud(background_color ='white',
                    stopwords = stopwords,
                    random_state = 42).generate(text)

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud)
    plt.title(params.title, fontdict={
        'size': params.size,
        'color': params.color
    })
    plt.axis("off")
    plt.tight_layout()
    plt.show()
def split_data(func):
    def wrapper(*args, **kwargs):
        X = func(*args, **kwargs)
        X_train, X_test, y_train, y_test = train_test_split(X, args[1], test_size=0.2, random_state=42)
        print(X_train.shape, y_train.shape)
        return X_train, X_test, y_train, y_test
    return wrapper

def fit_model(func):
    def wrapper(*args, **kwargs):
        clf = args[3]
        X_train, X_test, y_train, y_test= func(*args, **kwargs)
        
        clf.fit(X_train, y_train)
        return X_test, y_test, clf
    return wrapper

def fit_vectorizer(func):
    def wrapper(*args, **kwargs):
        corpus = args[0]
        vectorizer = args[2]
        X = vectorizer.fit_transform(corpus)
        print("Fitting vectorizer...", X.shape)
        display_stats(X, vectorizer, 10)
        return X
    return wrapper

def display_stats(X, vectorizer, N):
    tfs = np.asarray(X.sum(axis=0)).ravel()
    mask_inds = (-tfs).argsort()[:N]

    vocab_values = list(vectorizer.vocabulary_.values())
    terms = list(vectorizer.vocabulary_.keys())

    labels = []
    for i, j in enumerate(mask_inds):
        labels.append(terms[vocab_values.index(j)])

    plot_barh((labels, tfs[mask_inds]), title='Most occurred words', N=N, figsize=(8, 6))

def score_model(func):
    def wrapper(*args, **kwargs):
        X_test, y_true, clf = func(*args, **kwargs)
        y_pred = clf.predict(X_test)
        
        TP = np.sum(y_pred[np.where(y_true.ravel() == 1)[0]])
        TN = len(np.where(y_true.ravel() == 0)[0]) - np.sum(y_pred[np.where(y_true.ravel() == 0)[0]])
        FP = np.sum(y_pred[np.where(y_true.ravel() == 0)[0]])
        FN = len(np.where(y_true.ravel() == 1)[0]) - np.sum(y_pred[np.where(y_true.ravel() == 1)[0]])
        
        accuracy = (TP + TN) / len(y_true.ravel())
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        
        response = (accuracy, precision, recall, f1_score)
        plot_barh((['accuracy', 'precision', 'recall', 'F1'], response), N=4, figsize=(8, 4))
        return response
    return wrapper

@score_model
@fit_model
@split_data
@fit_vectorizer
def run_pipeline(raw_documents, target, vectorizer, model):
    pass 
def create_vocab(docs):
    vocab = dict()
    for tokens in docs:
        for token in tokens:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
    return vocab

def find_docs_matching_pattern(docs, search):
    occurrences = docs[docs.apply(len) > 0]
    occurrences = occurrences.apply(lambda x: np.sum([search in x]))
    pos = occurrences.index.values
    return train_df.loc[pos[np.where(occurrences.values > 0)[0]], 'question_text']
!ls "../input"
train_df = read_csv("../input/train.csv")
print()
test_df = read_csv("../input/test.csv")
print()
submission_df = read_csv("../input/sample_submission.csv")
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Train', 'Test']
sizes = [len(train_df), len(test_df)]
colors = ['#BB1AF7', "#20DAFF"]

patches, texts, autotexts = ax.pie(sizes, explode=(0, 0.1), labels=labels, colors=colors, autopct='%1.1f%%')
for text in texts:
    text.set_fontsize(20)
ax.axis('equal')
plt.show()
cloud_data = prepare_cloud_data(train_df.loc[train_df.target == 1, 'question_text'])

plot_wordcloud(cloud_data, set(), CloudParams('Insincere', 30, 'b'))
cloud_data = prepare_cloud_data(train_df.loc[train_df.target == 0, 'question_text'][:10000])

plot_wordcloud(cloud_data, set(), CloudParams('Sincere', 30, 'b'))
puncts = train_df.question_text.apply(lambda doc: [item for item in doc.split() if item in punctuation])
punct_vocabulary = create_vocab(puncts)
punct_vocabulary = sorted(punct_vocabulary.items(), key=lambda x: x[1], reverse=True)

plot_barh(([i[0] for i in punct_vocabulary], [i[1] for i in punct_vocabulary]), title='Puncts', N=10, figsize=(8, 6))
ALL_PUNCTS = list(set([i[0] for i in punct_vocabulary]))
np.asarray(ALL_PUNCTS)
caps = train_df.question_text.apply(lambda row: [token for token in row.split() if re.match(r"\b[A-Z]{3,}\b", token) is not None])
fig, ax = plt.subplots(figsize=(5, 4))
index = np.arange(2)
rects = ax.bar(index, [np.sum(caps.apply(len) == 0), np.sum(caps.apply(len) == 1)], width=0.35, color='m')
ax.set_xticks(index)
ax.set_xticklabels(['NO CAPS', 'HAS CAPS'])
ax.set_title('# CAPS', fontdict={'size': 15})
plt.show()
caps.apply(len).value_counts().sort_index()[::-1][:5]
for i in range(3):
    print(train_df.loc[np.random.choice(np.where(caps.apply(len) >= 10)[0]), 'question_text'])
digits = train_df.question_text.apply(lambda row: [token for token in row.split() if token.isdigit()])

dlens = train_df.question_text.apply(lambda row: np.sum([len(token) for token in row.split() if token.isdigit()]))
dlens.value_counts().sort_index()[:5]
digits.apply(len).value_counts().sort_index()[::-1][:5]
years = train_df.question_text.apply(lambda row: [token for token in row.split() if re.match(r"\b20\d{2}$\b", token) is not None])

years = years[years.apply(len) > 0]
row = []
col = []
data = []

vocabulary = defaultdict(int)
vocabulary.default_factory = vocabulary.__len__

for i, row in enumerate(years):
    feature_counter = {}
    for token in row:
        feature_indx = vocabulary[token]
        if feature_indx not in feature_counter:
            feature_counter[feature_indx] = 1
        else:
            feature_counter[feature_indx] += 1

vocabulary
def replace_year(doc):
    doc = re.sub(r"19[0-9][0-9]", r"19_yy", doc)
    return doc

def replace_number_abbrv(doc):
    doc = re.sub(r"\b(\d{1,2})000\b", r"\1k", doc)
    return doc

def clean_text(doc):
    doc = replace_year(doc)
#     doc = replace_number_abbrv(doc)
    return doc
filtered = train_df.copy()
filtered['cleaned_text'] = filtered.question_text.apply(lambda doc: clean_text(doc))
stoplist = set('and'.split())
vectorizer = CountVectorizer(max_features=10000, min_df=2, stop_words=stoplist)
lr = LogisticRegression()
stoplist
indx = np.random.choice(filtered.index.values)
print(filtered.loc[indx, 'question_text'])
print()
print(vectorizer.build_tokenizer()(filtered.loc[indx, 'cleaned_text']))
response = run_pipeline(filtered.cleaned_text, filtered.target, vectorizer, lr)
response
#(0.954238683127572, 0.6929009294047854, 0.4419778002018164, 0.5396996534462841)
print(vectorizer.get_feature_names()[:200])
