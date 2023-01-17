# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re # Regular expression
import nltk # text processing
from collections import namedtuple, defaultdict
import numbers
from wordcloud import WordCloud

# Plotting library
import matplotlib.pyplot as plt
% matplotlib inline

from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer # text processing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, log_loss

import scipy.sparse as sp
from nltk.stem import WordNetLemmatizer 

import gc

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
class BoW(object):
    '''
    Calculates frequencies of tokens across documents using Bag Of Words from scikit-learn
    '''                 
    def __init__(self):
        self.vect = None
        self.bow = None
        
    def sort_vocabulary(self, order=-1):
        '''
        Sorts the vocabulary in a given order
        
        Parameters
        -------------
        sort_order: {integer}, sorts the vocabulary.If order = -1, sort in descending
        '''
        tfs = np.array(self.bow.sum(axis=0)).ravel()
        sort_indx = (order * tfs).argsort()
        
        labels = list()
        values = list()
        terms = list(self.vect.vocabulary_.keys())
        indices = list(self.vect.vocabulary_.values())
        for index in sort_indx:
            labels.append(terms[indices.index(index)])
            values.append(tfs[index])
            
        return labels, values
        
    def fit(self, corpus, token_pattern):
        self.vect = CountVectorizer(token_pattern=token_pattern)
        self.bow = self.vect.fit_transform(corpus)
        
    def clean(self):
        self.vect = None
        self.bow = None
        
def search_pattern(token_pattern, corpus):
    pat = re.compile(token_pattern)
    matches = corpus.apply(lambda doc: re.findall(pat, doc))
    return matches
        
def create_vocab(tokens, ascending=None):
    '''
    Creates a vocabulary (key, value) pairs of words to frequencies
    
    Parameters
    --------------
    token_list : {array of array} - list of list of tokens
    '''
    
    counter = defaultdict()
    counter.default_factory = counter.__len__

    for doc in tokens:
        for token in doc:
            counter[token] += 1
            
    counter = dict(counter)
    if ascending is None:
        return counter
    
    if ascending is False:
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return sorted(counter.items(), key=lambda x: x[1])

def plot_barchart(labels, values, chart_params, figsize=(10, 4), horizontal=False):
    '''
    Plots a barchart
    
    Parameters
    -----------------
    labels : {array}, labels on the x-axis
    values : {array}, values on the y-axis
    chart_params : {namedtuple}, chart configurations
    '''
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(values))
    if horizontal is False:
        ax.bar(y_pos, values, chart_params.width, color=chart_params.colors)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(labels)
    else:
        ax.barh(y_pos, values, chart_params.width, color=chart_params.colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
    
    ax.set_title(chart_params.title, fontdict={'size': chart_params.title_size})
    plt.show()

def gen_cloud_data(corpus):
    '''
    Generates a word cloud data
    
    Parameters
    ------------
    corpus : {array} - Array of array of tokens
    '''
    digit_cloud = ' '
    for row in corpus:
        digit_cloud = digit_cloud + '_' + re.sub(r'\s+', '_', row) + ' '
    return digit_cloud

def plot_wordcloud(cloud_data, stops, params, figsize=(9, 7)):
    '''
    Plots a word cloud from array of documents
    '''
    all_stops = stop_words.ENGLISH_STOP_WORDS.union(stops)
    wordcloud = WordCloud(background_color ='white',
                    stopwords = all_stops,
                    max_words = 400,
                    max_font_size = 120, 
                    random_state = 42).generate(cloud_data)

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud)
    plt.title(params.title, fontdict={
        'size': params.title_size,
        'color': 'green'
    })
    plt.axis("off")
    plt.show()
    
def print_statements_from_index(indx_list, N=5):
    for i in range(N):
        indx = np.random.choice(indx_list)
        print('Index -> {}'.format(indx))
        print(train_df.loc[indx, 'question_text'])
    
# Declare namedtuples

ChartParams = namedtuple('ChartParams', ['title', 'title_size'])

BarChartParams = namedtuple('BarChart', ['title', 'title_size', 'width', 'colors'])
def get_uppers(tokens):
    return tokens.apply(lambda row: np.array(row)[np.where([token.isupper() for token in row])[0]])

def get_digits(tokens):
    return tokens.apply(lambda row: np.array(row)[np.where([token.isdigit() for token in row])[0]])

def get_alphanumerics(tokens):
    return tokens.apply(lambda row: [token for token in row if re.match(r"([a-zA-Z]+\d+|\d+[a-zA-Z]+)", token) is not None])

def get_currency(tokens):
    return tokens.apply(lambda row: [token for token in row if re.match(r"[$]\d\d+[a-z]+$", token) is not None])

def get_hyperlinks(tokens):
    return tokens.apply(lambda row: [token for token in row if re.match(r'\bhttps?[://].*[a-zA-Z0-9-&_.]$\b', token) is not None])

def get_masked_tokens(tokens, low=None, high=None):
    if high is None and low is None:
        return tokens
    
    masked = np.zeros(len(tokens), dtype=bool)
    
    if low is not None:
        masked = tokens.apply(len) > low
    if high is not None:
        masked &= tokens.apply(len) <= high
        
        
    return tokens[np.where(masked)[0]]
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

print("Total records in train data = {0}".format(train_df.shape[0]))
print("Total records in test data = {0}".format(test_df.shape[0]))
tokens = train_df.question_text.apply(lambda doc: doc.split())
uppers = get_uppers(tokens)
digits = get_digits(tokens)
alphanums = get_alphanumerics(tokens)
currencies = get_currency(tokens)
N = 10
num_pat = r'\b19\d{2}|20\d{2}\b'
bow = BoW()
bow.fit(train_df.question_text, num_pat)
labels, values = bow.sort_vocabulary()
plot_barchart(labels[:N], values[:N], BarChartParams('Top {} numbers'.format(N), 20, 0.5, 'm'))

bow = None
gc.collect()
cloud_data = gen_cloud_data(np.hstack(get_masked_tokens(digits)))
plot_wordcloud(cloud_data, set(), ChartParams('Most common numbers', 30))
gc.collect()
N = 10
alphaVocab = create_vocab(alphanums, ascending=True)[:N]
plot_barchart([i[0] for i in alphaVocab], [i[1] for i in alphaVocab], BarChartParams('Top {} alphanumerics'.format(N), 20, 0.5, 'r'))
N = 10
bar_data = create_vocab(get_masked_tokens(currencies, 0), ascending=True)[:10]
plot_barchart([i[0] for i in bar_data], [i[1] for i in bar_data], BarChartParams('Top {} currencies'.format(N), 20, 0.5, '#d8db2e'))

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 7))

ax[0][0].hist(get_masked_tokens(tokens, 3, 10).apply(len) , color='#4f2951')
ax[0][1].hist(get_masked_tokens(uppers, 4, 10).apply(len), color='#13b230')
ax[1][0].hist(get_masked_tokens(digits, 1, 10).apply(len), color='#3c2ed3')
ax[1][1].hist(get_masked_tokens(alphanums, 1, 10).apply(len), color='#d8db2e')
plt.show()
hyperlinks_series = get_hyperlinks(tokens)
hyperlinks = get_masked_tokens(hyperlinks_series, 0)
hyperlinks_df = pd.DataFrame({
    'links': hyperlinks
})
hyperlinks_df['domain'] = hyperlinks_df.links.apply(lambda links: [re.match(r"(https?\:\/\/).*", link).group(1) for link in links])
hyperlinks_df['url_first'] = hyperlinks_df.links.apply(lambda links: [re.match(r"https?\:\/\/([a-zA-Z0-9.-]+)\/?.*", link).group(1) for link in links if re.match(r"https?\:\/\/([\w.])\/?", link) is not None])
hyperlinks_df['query_params'] = hyperlinks_df.links.apply(lambda links: [re.match(r"\bhttps?\:\/\/.*[?](.*)\b", link).group(1) for link in links if re.match(r"\bhttps?\:\/\/.*[?](.*)\b", link) is not None])

N = 60
bar_data = create_vocab(hyperlinks_df['url_first'], ascending=True)[30:N]
plot_barchart([i[0] for i in bar_data], [i[1] for i in bar_data], BarChartParams(''.format(N), 20, 0.5, '#d8db2e'), figsize=(10, 8), horizontal=True)

def replace_urls(doc, url_pattern):
    return re.sub(url_pattern, r'\1', doc)

def replace_months(doc):
    return re.sub(r'(jan|feb|march|april|may|june|july|august|september|october|november|december)', r'month', doc)

def preprocess(doc):
    doc = replace_months(doc)
    doc = re.sub(r'(youtu)[.](be)', r'\1\2', doc)
    doc = re.sub(r'i[.](imgur)', r'\1', doc)
    doc = re.sub(r'\b\s{2}\b', r'', doc)
    return doc
train_df['preprocessed_text'] = train_df.question_text.apply(lambda doc: preprocess(doc))
# Custom stop words list

X = train_df.preprocessed_text
y = train_df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

STOPS = set(['to'])

def filter_tokens(tokens):
    
#     tokens = [t for t in tokens if t not in STOPS]
    original_tokens = list(tokens)
    return tokens

class CustomVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(CustomVectorizer, self).build_tokenizer()
        return lambda doc: list(filter_tokens(tokenize(doc)))
MAX_FEATURES = 10000
count_vect = CustomVectorizer(min_df=5, max_features=MAX_FEATURES, lowercase=False)
count_vect.fit(X_train)
dtm = count_vect.transform(X_train)

clf = LogisticRegression()
clf.fit(dtm, y_train)
def sort_bow_vocab(vect, dtm, order=-1):
    tfs = np.asarray(dtm.sum(axis=0)).ravel()
    sorted_indices = (order * tfs).argsort()

    terms = list(vect.vocabulary_.keys())
    indices = list(vect.vocabulary_.values())
    
    labels = []
    values = []
    for i in sorted_indices:
        values.append(tfs[i])
        labels.append(terms[indices.index(i)])
        
    return labels, values

labels, values = sort_bow_vocab(count_vect, dtm, order=1)

plot_barchart(labels[:10], values[:10], BarChartParams('Top {} words'.format(10), 20, 0.5, 'm'))
X_test_vectors = count_vect.transform(X_test)
predictions = clf.predict(X_test_vectors)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('AUC: ', roc_auc_score(y_test, predictions))
print('Accuracy: ', accuracy_score(y_test, predictions))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(false_positive_rate, true_positive_rate, color='green', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
# AUC:  0.7077320472912008
# Accuracy:  0.9541437656169885
!ls -l '../input/embeddings'
# from gensim.models import KeyedVectors

# EMBEDDINGS = '../input/embeddings/'
# embeddings_index = KeyedVectors.load_word2vec_format(os.path.join(EMBEDDINGS, 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'), binary=True)
# test_predictions.loc[:, 'qid'] = test_df.loc[:, 'qid']
# test_predictions.to_csv('submission.csv', index=False)