# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import warnings  
warnings.filterwarnings('ignore')

# importing packages
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# sklearn packages
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

# nltk packages
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from string import punctuation
import unidecode
data = pd.read_csv('../input/winemag-data_first150k.csv')
data.head(5)
data_sel = data.drop(['Unnamed: 0','designation','points','region_2',], axis = 1)
data_sel.shape
data_single = data_sel.drop_duplicates('description')
data_single.shape
data_single = data_single.dropna(subset=['description', 'variety', 'price'])
data_single.head()
data_single.describe(include='all')
for col in ['variety', 'description', 'province', 'region_1', 'winery', 'country']:
    data_single[col] = data_single[col].str.lower()

def unidecode_text(text):
    try:
        #pdb.set_trace()
        text = unidecode.unidecode(text)
    except:
        pass
    return text

for col in ['description', 'variety', 'province', 'winery']:
    data_single[col] = data_single.apply(lambda row: unidecode_text(row[col]), axis=1)
data_single.variety.value_counts()
filtered_name = ['red blend', 'portuguese red', 'white blend', 'sparkling blend', 'champagne blend', 
                 'portuguese white', 'rose', 'bordeaux-style red blend', 'rhone-style red blend',
                 'bordeaux-style white blend', 'alsace white blend', 'austrian red blend',
                 'austrian white blend', 'cabernet blend', 'malbec blend', 'portuguese rose',
                 'portuguese sparkling', 'provence red blend', 'provence white blend',
                 'rhone-style white blend', 'tempranillo blend', 'grenache blend',
                 'meritage' # beaurdaux blend
                ]
data_filtered = data_single.copy()
data_filtered = data_filtered[~data_filtered['variety'].isin(filtered_name)]
def correct_grape_names(row):
    regexp = [r'shiraz', r'ugni blanc', r'cinsaut', r'carinyena', r'^ribolla$', r'palomino', r'turbiana', r'verdelho', r'viura', r'pinot bianco|weissburgunder', r'garganega|grecanico', r'moscatel', r'moscato', r'melon de bourgogne', r'trajadura|trincadeira', r'cannonau|garnacha', r'grauburgunder|pinot grigio', r'pinot noir|pinot nero', r'colorino', r'mataro|monastrell', r'mourv(\w+)']
    grapename = ['syrah', 'trebbiano', 'cinsault', 'carignan', 'ribolla gialla', 'palomino','verdicchio', 'verdejo','macabeo', 'pinot blanc', 'garganega', 'muscatel', 'muscat', 'muscadet', 'treixadura', 'grenache', 'pinot gris', 'pinot noir', 'lambrusco', 'mourvedre', 'mourvedre']
    f = row
    for exsearch, gname in zip(regexp, grapename):
        f = re.sub(exsearch, gname, f)
    return f

name_pairs = [('spatburgunder', 'pinot noir'), ('garnacha', 'grenache'), ('pinot nero', 'pinot noir'),
              ('alvarinho', 'albarino'), ('assyrtico', 'assyrtiko'), ('black muscat', 'muscat hamburg'),
              ('kekfrankos', 'blaufrankisch'), ('garnacha blanca', 'grenache blanc'),
              ('garnacha tintorera', 'alicante bouschet'), ('sangiovese grosso', 'sangiovese')
             ]
data_corrected = data_filtered.copy()
data_corrected['variety'] = data_corrected['variety'].apply(lambda row: correct_grape_names(row))
for start, end in name_pairs:
    data_corrected['variety'] = data_corrected['variety'].replace(start, end) 

len(data_corrected.variety.value_counts())
data_reduced = data_corrected.groupby('variety').filter(lambda x: len(x) > 200)
data_reduced.shape
grapes = list(np.unique(data_reduced.variety.value_counts().index.tolist()))
len(grapes)
data_reduced.variety.value_counts()
colour_map = {'aglianico': 'red', 'albarino': 'white', 'barbera': 'red', 'cabernet franc': 'red',
              'cabernet sauvignon': 'red', 'carmenere': 'red', 'chardonnay': 'white', 'chenin blanc': 'white',
              'corvina, rondinella, molinara': 'red', 'gamay': 'red', 'garganega': 'white', 
              'gewurztraminer': 'white', 'glera': 'white', 'grenache': 'red', 'gruner veltliner': 'white',
              'malbec': 'red', 'merlot': 'red', 'mourvedre': 'red', 'muscat': 'white', 'nebbiolo': 'red',
              "nero d'avola": 'red', 'petite sirah': 'red', 'pinot blanc': 'white', 'pinot gris': 'white',
              'pinot noir': 'red', 'port': 'red', 'prosecco': 'white', 'riesling': 'white', 'sangiovese': 'red',
              'sauvignon blanc': 'white', 'syrah': 'red', 'tempranillo': 'red', 'torrontes': 'white', 
              'verdejo': 'white', 'viognier': 'white', 'zinfandel': 'white'
             }
kaggle_input = data_reduced.copy()
kaggle_input['colour'] = kaggle_input.apply(lambda row: colour_map[row['variety']], axis=1)
colour_dummies = pd.get_dummies(kaggle_input['colour'])
kaggle_input = kaggle_input.merge(colour_dummies, left_index=True, right_index=True)
kaggle_input.reset_index(inplace=True)
kaggle_input.head()
# stop words for countries
stop_country = list(np.unique(kaggle_input.country.dropna().str.lower().tolist()))

#stop words for province
stop_province = list(np.unique(kaggle_input.province.dropna().str.lower().tolist()))

#stop words for winery
stop_winery = list(np.unique(kaggle_input.winery.dropna().str.lower().tolist()))

# defining stopwords: using the one that comes with nltk + appending it with words seen from the above evaluation
stop_words = stopwords.words('english')
stop_append = ['.', ',', '`', '"', "'", '!', ';', 'wine', 'fruit', '%', 'flavour', 'aromas', 'palate']
stop_words1 = stop_words + stop_append + grapes + stop_country + stop_province + stop_winery
# list of word types (nouns and adjectives) to leave in the text
defTags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']#, 'RB', 'RBS', 'RBR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# functions to determine the type of a word
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

# transform tag forms
def penn_to_wn(tag):
    if is_adjective(tag):
        return nltk.stem.wordnet.wordnet.ADJ
    elif is_noun(tag):
        return nltk.stem.wordnet.wordnet.NOUN
    elif is_adverb(tag):
        return nltk.stem.wordnet.wordnet.ADV
    elif is_verb(tag):
        return nltk.stem.wordnet.wordnet.VERB
    return nltk.stem.wordnet.wordnet.NOUN
    
# lemmatizer + tokenizer (+ stemming) class
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        # we define (but not use) a stemming method, uncomment the last line in __call__ to get stemming tooo
        self.stemmer = nltk.stem.SnowballStemmer('english') 
    def __call__(self, doc):
        # pattern for numbers | words of length=2 | punctuations | words of length=1
        pattern = re.compile(r'[0-9]+|\b[\w]{2,2}\b|[%.,_`!"&?\')({~@;:#}+-]+|\b[\w]{1,1}\b')
        # tokenize document
        doc_tok = word_tokenize(doc)
        #filter out patterns from words
        doc_tok = [x for x in doc_tok if x not in stop_words1]
        doc_tok = [pattern.sub('', x) for x in doc_tok]
        # get rid of anything with length=1
        doc_tok = [x for x in doc_tok if len(x) > 1]
        # position tagging
        doc_tagged = nltk.pos_tag(doc_tok)
        # selecting nouns and adjectives
        doc_tagged = [(t[0], t[1]) for t in doc_tagged if t[1] in defTags]
        # preparing lemmatization
        doc = [(t[0], penn_to_wn(t[1])) for t in doc_tagged]
        # lemmatization
        doc = [self.wnl.lemmatize(t[0], t[1]) for t in doc]
        # uncomment if you want stemming as well
        #doc = [self.stemmer.stem(x) for x in doc]
        return doc
vec_tdidf = TfidfVectorizer(ngram_range=(1,1), analyzer='word', #stop_words=stop_words1, 
                                               norm='l2', tokenizer=LemmaTokenizer())
clf = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)
class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None, *parg, **kwarg):
        return self

    def transform(self, X):
        # returns the input as a string
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # returns the input as a dataframe
        return X[[self.key]]
def print_stats(preds, target, labels, sep='-', sep_len=40, fig_size=(10,8)):
    print('Accuracy = %.3f' % metrics.accuracy_score(target, preds))
    print(sep*sep_len)
    print('Classification report:')
    print(metrics.classification_report(target, preds))
    print(sep*sep_len)
    print('Confusion matrix')
    cm=metrics.confusion_matrix(target, preds)
    cm = cm / np.sum(cm, axis=1)[:,None]
    sns.set(rc={'figure.figsize':fig_size})
    sns.heatmap(cm, 
        xticklabels=labels,
        yticklabels=labels,
           annot=True, cmap = 'YlGnBu')
    plt.pause(0.05)
text = Pipeline([
                ('selector', TextSelector(key='description')),
                ('vectorizer', vec_tdidf)
                ])
#pipelines of colour features
red = Pipeline([
                ('selector', NumberSelector(key='red')),
                ])
white = Pipeline([
                ('selector', NumberSelector(key='white')),
                ])
feats = FeatureUnion([('description', text),
                      ('red', red),
                      ('white', white)
                      ])
pipe = Pipeline([('feats', feats),
                 ('clf',clf)
                 ])
pipe.named_steps['clf'].get_params()
# split the data into train and test
combined_features = ['description', 'white', 'red']
target = 'variety'

X_train, X_test, y_train, y_test = train_test_split(kaggle_input[combined_features], kaggle_input[target], 
                                                    test_size=0.33, random_state=42, stratify=kaggle_input[target])

# definition of parameter grid to scan through
param_grid = {
     'clf__n_estimators': [50,100,300]
#    'clf__colsample_bytree': [0.6,0.8,1]
#    'clf__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1]
}
# grid search cross validation instantiation
grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid, 
                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)
#hyperparameter fitting
grid_search.fit(X_train, y_train)
grid_search.cv_results_['mean_train_score']
grid_search.cv_results_['mean_test_score']
grid_search.best_params_
clf_test = grid_search.best_estimator_
# test stats
preds = clf_test.predict(X_test)
print_stats(y_test, preds, clf_test.classes_)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline as imbPipeline
clf = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7, 
                    n_estimators=300)
sm = SMOTE(random_state=42, sampling_strategy='auto')
pipe = imbPipeline([('feats', feats),
                    ('smote', sm),
                    ('clf',clf)
                 ])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
print_stats(y_test, preds, pipe.classes_)