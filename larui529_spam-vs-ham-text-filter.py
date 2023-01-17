# import some necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

%matplotlib inline 

import nltk

from nltk.corpus import stopwords

import re

import string

from nltk.corpus import stopwords

from collections import Counter

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, Lasso, Ridge

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.kernel_ridge import KernelRidge

import lightgbm as lgb

from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb

from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# now let's import data in pandas dataframe

df = pd.read_csv('../input/spam.csv',usecols = [0,1],encoding='latin-1' )

df.rename(columns = {'v1':'Category','v2': 'Message'}, inplace = True)
df.head()
df.groupby('Category').describe()
category_count = pd.DataFrame()

category_count['count'] = df['Category'].value_counts()
fig, ax = plt.subplots(figsize = (12, 6))

sns.barplot(x = category_count.index, y = category_count['count'], ax = ax)

ax.set_ylabel('count', fontsize = 15)

ax.set_xlabel('category',fontsize = 15)

ax.tick_params(labelsize=15)
spam_df = df[df['Category'] == 'spam'] #create sub-dataframe of spam text

ham_df = df[df['Category'] == 'ham'] #sub-dataframe of ham text
stop_words = set(stopwords.words('english'))

def wordCount(text):

    try:

        text = text.lower()

        regex = re.compile('['+re.escape(string.punctuation) + '0-9\\r\\t\\n]') 

        txt = regex.sub(' ',text)  #remove punctuation

        words = [w for w in txt.split(' ')\

                if not w in stop_words and len(w)>3] # remove stop words and words with length smaller than 3 letters

        return len(words)

    except:

        return 0
spam_df['len'] = spam_df['Message'].apply(lambda x: len([w for w in x.split(' ')]))

ham_df['len'] = ham_df['Message'].apply(lambda x: len([w for w in x.split(' ')]))

spam_df['processed_len'] = spam_df['Message'].apply(lambda x: wordCount(x))

ham_df['processed_len'] = ham_df['Message'].apply(lambda x: wordCount(x))
xmin = 0

xmax = 50

print ('spam length info')

print (spam_df[['len', 'processed_len']].describe())

print ('ham length info')

print (ham_df[['len', 'processed_len']].describe())

fig, ((ax,ax1),(ax2,ax3)) = plt.subplots (2,2,figsize = (12,9))

spam_df['len'].plot.hist(bins = 20, ax = ax, edgecolor = 'white', color = 'orange')

spam_df['processed_len'].plot.hist(bins = 20, ax = ax1, edgecolor = 'white', color = 'orange')

ham_df['len'].plot.hist(bins = 20, ax = ax2, edgecolor = 'white', color = 'blue')

ham_df['processed_len'].plot.hist(bins = 20, ax = ax3, edgecolor = 'white', color = 'blue')

ax.tick_params(labelsize = 15)

ax.set_xlabel('length of sentence', fontsize = 12)

ax.set_ylabel('spam_frequency', fontsize = 12)

ax.set_xlim([xmin,xmax])

ax1.tick_params(labelsize = 15)

ax1.set_xlabel('length of processed sentence', fontsize = 12)

ax1.set_ylabel('spam_frequency', fontsize = 12)

ax1.set_xlim([xmin,xmax])

ax2.tick_params(labelsize = 15)

ax2.set_xlabel('length of sentence', fontsize = 12)

ax2.set_ylabel('ham_frequency', fontsize = 12)

ax2.set_xlim([xmin,xmax])

ax3.tick_params(labelsize = 15)

ax3.set_xlabel('length of processed sentence', fontsize = 12)

ax3.set_ylabel('ham_frequency', fontsize = 12)

ax3.set_xlim([xmin,xmax])
def tokenize(text):

    exclude = set(string.punctuation)

    regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]') #remove punctuation

    text = regex.sub(' ', text)

    tokens = nltk.word_tokenize(text) # tokenize the text

    tokens = list(filter(lambda x: x.lower() not in stop_words, tokens)) # remove stop words

    tokens = [w.lower() for w in tokens if len(w) >=3] 

    tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]

    return tokens
spam_df['tokens'] = spam_df['Message'].map(tokenize)

ham_df['tokens'] = ham_df['Message'].map(tokenize)
spam_words = []

for token in spam_df['tokens']:

    spam_words = spam_words + token #combine text in different columns in one list

ham_words = []

for token in ham_df['tokens']:

    ham_words += token
spam_count = Counter(spam_words).most_common(10)

ham_count = Counter(ham_words).most_common(10)
spam_count_df = pd.DataFrame(spam_count, columns = ['word', 'count'])

ham_count_df = pd.DataFrame(ham_count, columns = ['word', 'count'])
spam_count

fig, (ax,ax1) = plt.subplots(1,2,figsize = (18, 6))

sns.barplot(x = spam_count_df['word'], y = spam_count_df['count'], ax = ax)

ax.set_ylabel('count', fontsize = 15)

ax.set_xlabel('word',fontsize = 15)

ax.tick_params(labelsize=15)

ax.set_title('spam top 10 words', fontsize = 15)

sns.barplot(x = ham_count_df['word'], y = ham_count_df['count'], ax = ax1)

ax1.set_ylabel('count', fontsize = 15)

ax1.set_xlabel('word',fontsize = 15)

ax1.tick_params(labelsize=15)

ax1.set_title('ham top 10 words', fontsize = 15)

spam_words_str = ' '.join(spam_words)

ham_words_str = ' '.join(ham_words)
spam_word_cloud = WordCloud(width = 600, height = 400, background_color = 'white').generate(spam_words_str)

ham_word_cloud = WordCloud(width = 600, height = 400,background_color = 'white').generate(ham_words_str)
fig, (ax, ax2) = plt.subplots(1,2, figsize = (18,8))

ax.imshow(spam_word_cloud)

ax.axis('off')

ax.set_title('spam word cloud', fontsize = 20)

ax2.imshow(ham_word_cloud)

ax2.axis('off')

ax2.set_title('ham word cloud', fontsize = 20)

plt.show()
df['tokens'] = df['Message'].map(tokenize)
def text_join(text):

    return " ".join(text)

df['text'] = df['tokens'].apply(text_join)
tv = TfidfVectorizer('english')

features = tv.fit_transform(df['text'])

target = df.Category.map({'ham':0, 'spam':1})
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, Lasso, Ridge

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.kernel_ridge import KernelRidge

import lightgbm as lgb

from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb

from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
n_folds = 5

def f1_cv(model):

    kf = KFold(n_folds, shuffle = True, random_state = 29).get_n_splits(features)

    f1 = cross_val_score(model, features, target, scoring = 'f1', cv = kf )

    return (f1)
svc = SVC(kernel = 'sigmoid', gamma = 1.0)

rfc = RandomForestClassifier(n_estimators = 31, random_state = 32)
GBoost = GradientBoostingClassifier( n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   random_state =5)
model_lgb = lgb.LGBMClassifier(

                              objective='binary',num_leaves=5,

                              learning_rate=0.05, n_estimators=4420,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 1)
mnb = MultinomialNB(alpha = .2)
score = f1_cv(svc)

print ('\nSVC score: {:4f}({:4f})\n'.format(score.mean(), score.std()))
score = f1_cv(rfc)

print ('\nRandomForest score: {:4f}({:4f})\n'.format(score.mean(), score.std()))
score = f1_cv(mnb)

print ('\nMultinomial NB score: {:4f}({:4f})\n'.format(score.mean(), score.std()))
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
def classifier(clf, X_train, y_train):    

    clf.fit(X_train, y_train)

def predictor(clf, X_test):

    return (clf.predict(X_test))
clf = {'SVC':svc, 'RandomForest':rfc,  'MultinomialNB': mnb}

preds = []

for key, value in clf.items():

    #print(key)

    classifier(value, X_train, y_train)

    pred = predictor(value,X_test)

    preds.append((key, [accuracy_score(y_test,pred)]))
preds