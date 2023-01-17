# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined bty the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from __future__ import print_function

import os
import json
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy

import time, random, math
import re  #regular expression
import operator
from collections import defaultdict, Counter, OrderedDict, namedtuple
import numbers


import nltk#natural language processing
import six
import scipy.sparse as sp
from operator import itemgetter
import string

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

%matplotlib inline
matplotlib.rc('axes', facecolor = 'white')

import seaborn as sns
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier as RFC, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, log_loss

print(os.listdir("../input"))
def timer(func):
    def wrapper(*args, **kwargs):
        ts = time.time()
        results = func(*args, **kwargs)
        te = time.time()
        print("Time to execute {} = {} seconds".format(func.__name__, te - ts))
        return results
              
    return wrapper

from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk_stemmer = PorterStemmer()
nltk_lemma = WordNetLemmatizer()


ChartParams = namedtuple('ChartParams', ['title', 'xticks', 'yticks', 'xlabel', 'ylabel', 'colors', 'is_horizontal', 'min_max'])
# @TODO, for future refence
# def currency(x, pos):
#     """The two args are the value and tick position"""
#     if x >= 1e6:
#         s = '${:1.1f}M'.format(x*1e-6)
#     else:
#         s = '${:1.0f}K'.format(x*1e-3)
#     return s

def formatNum(num):
    if isinstance(num, numbers.Integral):
        return num
    return "{0:.3f}".format(num)
    
def barplot(ax, data, chart_params):
    """
    Plots a horizontal or vertical barplot
    Parameters
    -------------
    ax - the figure axes
    """
    
    N = len(data)
    pos = np.arange(N)
    height=0.8
    #formatter = FuncFormatter(currency)

    if chart_params.is_horizontal:
        if len(chart_params.colors) > 0:
            rects = ax.barh(pos, data, align='center', height=height, color=chart_params.colors)
        else:
            rects = ax.barh(pos, data, align='center', height=height)

        ax.set_title(chart_params.title, size=14)
        ax.xaxis.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
        ax.set_yticks(pos)
        ax.set_yticklabels(chart_params.yticks)
        ax.set(xlim=[chart_params.min_max[0], chart_params.min_max[1]], xlabel=chart_params.xlabel, ylabel=chart_params.ylabel)
        
        for i, rect in enumerate(rects):
            rect_width = rect.get_width()
            xloc = rect_width if rect_width < 1 else int(rect_width)*0.98
            yloc = rect.get_y() + rect.get_height()/2.0
            ax.text(xloc, yloc, formatNum(data[i]), horizontalalignment='right', verticalalignment='center', color='white', weight='bold', clip_on=True)

            #ax.xaxis.set_major_formatter(formatter)
    else:
        rects = ax.bar(pos, data, width, color=colors)
        ax.set_xticks(pos)
        ax.set_xticklabels(chart_params.xticks)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
            
        for i, rect in enumerate(rects):
            xloc = rect.get_x() + 0.02
            yloc = rect.get_height()
            ax.text(xloc, yloc, str(data[i]), weight='bold', color='b')
            
def getMainIngredientsByCuisine(tfm, vectorizer, target):
    results = {}
    for cuisine in np.unique(train_df.cuisine):
        ids = np.where(target == cuisine)[0]
        data = tfm[ids,:].toarray()
        tfs = np.array(data.sum(axis=0)).ravel()
        indices = np.argsort(-tfs)
        names = []
        counts = []
        vocab = list(six.iteritems(vectorizer.vocabulary_))
        for i, index in enumerate(indices):
            names.append([tup[0] for tup in vocab if tup[1] == index][0])
            counts.append(tfs[index])
        results[cuisine] = zip(names, counts)
        
    return results

def getFrequencyDistribution(words_list, ascending=False):
    '''
        Calculate the occurrency of each token in a list
        Parameters
        ------------
        words_list : array
                    list of tokens occurring more than once
        ascending  : boolean, default=False
                    the order in which the results is sorted
    '''
    vocab = defaultdict()
    vocab.default_factory = vocab.__len__

    for tokens in words_list:
        if len(tokens) == 0:
            continue
        for token in tokens:
            vocab[token] += 1
        
    vocab = dict(vocab)
    terms = np.array(list(vocab.keys()))
    values = np.array(list(vocab.values()))
    flag = -1
    if ascending:
        flag = 1
    indices = np.argsort(flag*values)
    return list(zip(terms[indices], values[indices]))

def find_indices_of_word_groups(vect, words_list):
    words_indices = [vect.vocabulary_.get(w) for w in words_list]
    return words_indices
   
def get_vocab_subset(words_list, tfm, vect):
    words_indices = find_indices_of_word_groups(vect, words_list)
    all_indices = np.array([tup[1] for tup in sorted(six.iteritems(vect.vocabulary_))])
    kept_indices = [indx for indx in all_indices if indx not in words_indices]
    X1 = tfm[:, kept_indices]
    print(tfm.shape, X1.shape)
    return X1

def fit_and_score(X, target, estimator):
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)
    print("Training on {} features".format(X_train.shape))
    clf = copy.deepcopy(estimator)
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    predict_proba = clf.predict_proba(X_test)
    print(accuracy_score(y_test, predictions) * 100)
    print(log_loss(y_test, predict_proba))
    return clf

def append_column_with_TFM(X, data_col):
    '''
    Arguments
    -----------
    X - term frequency matrix
    data_col - 1D array to be appended
    '''
    if sp.issparse(X):
        X = X.toarray()
    return np.concatenate((X, data_col), axis=1)

def get_word_occurrence_by_cuisine(vect, tfm, target, words_list):
    '''
        Given a list of words finds the occurrence in each cuisine
        Parameters
        --------------
        target - the target values (list of cuisines)
        words_list - list of unique words in the vocabulary from countvectorizer
    '''
    words_indices = find_indices_of_word_groups(vect, words_list)
    unique_targets = np.unique(target)
    results = []
    labels = []
    for i, label in enumerate(unique_targets):
        cuisine_df = tfm[np.where(target == label)]
        tfs = cuisine_df.sum(axis=0)
        results.append([cuisine_df[:, indx].sum() for indx in words_indices])
        labels.append(label)
        
    return pd.DataFrame(results, columns=words_list, index=labels)

def plot_occurrence_across_cuisines(vect, tfm, word):
    df = get_word_occurrence_by_cuisine(vect, tfm, target, [word])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, wspace=.35, hspace=.35)
    chart_params = ChartParams('Count of ' + word, '', df[word].index.values, '', '', ['#15CBCB'], True, (0, max(df[word].values)))
    barplot(ax, df[word].values, chart_params)
    plt.show()
    
_uppercase_pat = re.compile(r'[A-Z]\w+')
    
def plotMainIngredients(tfm, vect, N):
    n_row = 5
    n_col = 4
    colors = ['#A66AF7', '#3290F5', '#EE1F12', "#8156A4"]
    ingredientsByCounts = getMainIngredientsByCuisine(tfm, vect, target)

    fig, axarr = plt.subplots(n_row, n_col, figsize=(5 * n_col, 10 * n_row), squeeze=False)
    #plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, wspace=.35, hspace=.35)

    for i in range(n_row):
        groups = list(train_df.groupby('cuisine').groups)[i * n_col:(i * n_col) + n_col]
        for indx, item in enumerate(groups):
            d = list(ingredientsByCounts[item])[:N]
            values = [item[1] for item in d]
            chart_params = ChartParams(item + ' (n=1)', '', [item[0] for item in d], '', '', colors[indx], True)
            barplot(axarr[i][indx], [item[1] for item in d], chart_params)

    plt.show()
 
def get_uniq_ingredients(vect):
    labels = []
    values = []
    for cuisine in np.unique(target):
        ids = np.where(target == cuisine)[0]
        d = tfm[ids]
        sum_words = np.array(d.sum(axis=0)).ravel()
        word_freq = [(word, sum_words[indx]) for word, indx in vect.vocabulary_.items()]
        word_freq = [tup for tup in word_freq if tup[1] > 0]
        labels.append(cuisine)
        values.append(len(word_freq))
        
    return labels, values

def get_variations_in_ingredients(ingredients_txt, word_list):
    labels = []
    matches = []
    for word in word_list:
        regex = re.compile(r"\b[a-z][a-z]+\s+" + word + r"\s*[a-z]*\b")
        matches.append(list(set(regex.findall(ingredients_txt))))
        labels.append(word)
    
    indices = np.where(np.array(list(map(len, matches))))[0]
    return np.array(labels).take(indices), np.array(matches).take(indices)

def sortListByFreq(vect, tfm, word_list, ascending=False):
    matrix_indices = [vect.vocabulary_.get(word) for word in word_list]
    matrix_indices = [num for num in matrix_indices if num is not None]
    sum_words = np.array(tfm[:, matrix_indices].sum(axis=0)).ravel()
    sign = -1
    if ascending:
        sign = 1
    sort_indices = np.argsort(sign * sum_words)
    values = np.array(sum_words)[np.argsort(-sum_words)]
    names = np.array(word_list)[np.argsort(-sum_words)]
    return list(zip(names, values))

def plotFrequency(ax, data, N, chartTitle):
    '''
    Plot frequency of tokens
    Parameters
    --------------
    data - array of tokens
    chartParams - parameters of chart,(e.g. xlabels, ylabels)
    figsize - width and height of chart figure
    N - total number of records to show in plot
    '''
    vocab = getFrequencyDistribution(data)
    labels = [item[0] for item in vocab][:N]
    counts = [item[1] for item in vocab][:N]
    chart_params = ChartParams(chartTitle, '', labels, '', '', [CHART_COLOR], True, (0, max(counts)))
    barplot(ax, counts, chart_params)
with open("../input/train.json", 'r') as file:
    data_train = json.load(file)
    
with open("../input/test.json", 'r') as file:
    data_test = json.load(file)
    
CHART_COLOR = '#B9D132'
train_df = pd.DataFrame(data_train)
test_df = pd.DataFrame(data_test)

clf = LogisticRegression(multi_class='warn', random_state=42)
train_df['concat_ingredients'] = train_df['ingredients'].apply(lambda row: ",".join(row))
test_df['concat_ingredients'] = test_df['ingredients'].apply(lambda row: ",".join(row))
corpus = train_df['concat_ingredients']
target = train_df['cuisine']
def clean_text(doc):
    doc = re.sub(r'\bmi\b', '', doc)
    doc = re.sub(r'\bfresh\b', '', doc)
    return doc
vect = CountVectorizer(preprocessor=clean_text, tokenizer=lambda doc: [word.strip() for word in doc.split(',')])
tfm = vect.fit_transform(corpus)
labels, values = get_uniq_ingredients(vect)
fig, ax = plt.subplots(figsize=(10, 8))
chart_params = ChartParams('#Unique ingredients', '', np.array(labels)[np.argsort(values)], '', '', ['#B9D132'], True, (0, max(values)))
barplot(ax, np.array(values)[np.argsort(values)], chart_params)
plt.show()
train_df['total_ingredients'] = train_df['ingredients'].apply(lambda row: len(row))
train_df['n_words'] = train_df['ingredients'].apply(lambda row: sum([len(w.split()) for w in row]))
train_df['n_uppercase_words'] = train_df['ingredients'].apply(lambda row: sum([1 if w[0].isupper() else 0 for w in row]))
train_df['n_digits'] = train_df['ingredients'].apply(lambda row: sum([1 if w[0].isdigit() else 0 for w in row]))

train_df['hyphen_ingredients'] = train_df['concat_ingredients'].apply(lambda row: re.compile(r'\b[a-z][a-z]+[-][a-z]+\b').findall(row))
train_df['negations'] = train_df['concat_ingredients'].apply(lambda row: re.compile(r'\b(non[\s-][a-z\s]+|no[\s-][a-z\s]+|not[\s-][a-z\s]+)\b').findall(row))
train_df['quoted_ingredients'] = train_df['concat_ingredients'].apply(lambda row: re.compile(r'\b([\w]+\'[\w]+)\b').findall(row))

train_df['punctuations'] = train_df['ingredients'].apply(lambda row: list(set(np.hstack([re.compile(r'\W').findall(w) for w in row]))))

nrows = 4
ncols = 1
fig, axarr = plt.subplots(nrows, ncols, figsize=(5, 8), squeeze=False)
plt.subplots_adjust(bottom=0, left=.001, right=.99, top=.90, wspace=.35, hspace=.35)

plotFrequency(axarr[0][0], train_df['hyphen_ingredients'].values, 5, '# Hyphenized ingredients')
plotFrequency(axarr[1][0], train_df['negations'].values, 5, '# Negation ingredients')
plotFrequency(axarr[2][0], train_df['quoted_ingredients'].values, 5, '# Quoted ingredients')
plotFrequency(axarr[3][0], train_df['punctuations'].values, 5, '# Punctuations')

plt.show()
# plot_occurrence_across_cuisines(vect, tfm, 'gluten free barbecue sauce')
def preprocess(doc):
    doc = re.sub(r"\s+"," ", doc, flags = re.I)  # remove spaces
    doc = re.sub(r'\W', ' ', doc, flags = re.I)  # remove non-words
    return doc

def preprocess_ngrams(doc):
    doc = re.sub(r"\s+"," ", doc, flags = re.I)  # remove spaces
    return doc
class CustomVectorizer(CountVectorizer):
    def build_tokenizer(self):
        return lambda doc: doc.split(",")
    
def tokenizer(doc):
    ingredients = doc.split(",")
    results = []
    results_extend = results.extend
    for ingredient in ingredients:
        words = ingredient.split()
        for word in words:
            if word[0].isdigit():
                continue
            if len(word) == 1:
                continue
            results_extend([word.lower()])
    return results

space_join = " ".join

def tokenizer_ngrams(doc):
    tokens = list(ingredient.strip() for ingredient in doc.split(","))
    results = []
    results_extend = results.extend
    for token in tokens:
        token = re.sub(r'\W', ' ', token, flags = re.I)
        w_splits = token.split()
        results_extend(w_splits)
        
        if len(w_splits) in [1, 2]:
            results_extend([token])
            
        if len(w_splits) in [3]:
            results_extend([space_join(w_splits[1:])])
            
        if len(w_splits) in [4]:
            results_extend([space_join(w_splits[1:3])])
            
        if len(w_splits) in [5, 6]:
            results_extend([token])
            
    return results

def tokenizer_all(doc):
    tokens = list(ingredient.strip() for ingredient in doc.split(","))
    unigrams = []
    for word in tokens:
        if len(word.split()) >= 1:
            unigrams.append(word.split()[-1])
        else:
            unigrams.append(word)
    
    results = []
    results_extend = results.extend
    for i, item in enumerate(unigrams):
        if i == len(unigrams) - 1:
            continue
        results_extend([space_join([unigrams[i], unigrams[i + 1]])])
    return results
vectorizer = CountVectorizer(preprocessor=preprocess, tokenizer=tokenizer, lowercase=False, stop_words=set(['brown', 'fresh', 'purple']))
X = vectorizer.fit_transform(corpus)

vectorizer_ngrams = CountVectorizer(preprocessor=preprocess_ngrams, tokenizer=tokenizer_ngrams, min_df=2, stop_words='english')
X_ngrams = vectorizer_ngrams.fit_transform(corpus)
# vectorizer_comb = CountVectorizer(tokenizer=tokenizer_all)
# X_comb = vectorizer_comb.fit_transform(corpus)

#cooking,condensed, converted, cokked, cooking, 
print(train_df.loc[np.random.choice(train_df.index), 'ingredients'])
train_df['no-oils'] = train_df.apply(lambda row: 1 if len([w for w in row['ingredients'] if len(re.split(r'\boil\b', w)) == 1]) == row['total_ingredients'] else 0, axis=1)


# print(vectorizer.get_feature_names())
lr_unigrams = fit_and_score(X, target, clf)

# 78.29498704860582
# 0.7706321294271727
lr_ngrams = fit_and_score(X_ngrams, target, clf)

# 78.58448880085326
# 0.7635537827858768
lr_all = fit_and_score(append_column_with_TFM(X_ngrams, train_df[['no-oils']].values), target, clf)

# 78.37117172024989
# 0.7713099458169165
test_df['total_ingredients'] = test_df['ingredients'].apply(lambda row: len(row))
test_df['no-oils'] = test_df.apply(lambda row: 1 if len([w for w in row['ingredients'] if len(re.split(r'\boil\b', w)) == 1]) == row['total_ingredients'] else 0, axis=1)

feature_vectors = vectorizer_ngrams.transform(test_df['concat_ingredients'])
submission = pd.DataFrame({"cuisine": lr_all.predict(append_column_with_TFM(feature_vectors, test_df[['no-oils']].values)), "id": test_df['id']})
submission.to_csv('submission1.csv', index=False)
