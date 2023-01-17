# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression,SGDClassifier

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

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
train.head()
# Target Mapping

mapping_target = {'EAP':0, 'HPL':1, 'MWS':2}

train = train.replace({'author':mapping_target})
train.head()
test_id = test['id']

target = train['author']
# function to clean data

import string

import itertools 

import re

from nltk.stem import WordNetLemmatizer

from string import punctuation



stops = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',

              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',

              'Is','If','While','This']

# punct = list(string.punctuation)

# punct.append("''")

# punct.append(":")

# punct.append("...")

# punct.append("@")

# punct.append('""')

def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):

    

    txt = str(text)

    

    txt = re.sub(r'[^A-Za-z\s]',r' ',txt)

    

     

    if lowercase:

        txt = " ".join([w.lower() for w in txt.split()])

        

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in stops])

    if stemming:

        st = PorterStemmer()

        txt = " ".join([st.stem(w) for w in txt.split()])

    

    if lemmatization:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])



    return txt
# def fraction_noun(row):

#     """function to give us fraction of noun over total words """

#     text = row['text']

#     text_splited = text.split(' ')

#     text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

#     text_splited = [s for s in text_splited if s]

#     word_count = text_splited.__len__()

#     pos_list = nltk.pos_tag(text_splited)

#     noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])

#     return (noun_count/word_count)



# def fraction_adj(row):

#     """function to give us fraction of adjectives over total words in given text"""

#     text = row['text']

#     text_splited = text.split(' ')

#     text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

#     text_splited = [s for s in text_splited if s]

#     word_count = text_splited.__len__()

#     pos_list = nltk.pos_tag(text_splited)

#     adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])

#     return (adj_count/word_count)



# def fraction_verbs(row):

#     """function to give us fraction of verbs over total words in given text"""

#     text = row['text']

#     text_splited = text.split(' ')

#     text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

#     text_splited = [s for s in text_splited if s]

#     word_count = text_splited.__len__()

#     pos_list = nltk.pos_tag(text_splited)

#     verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])

#     return (verbs_count/word_count)
# train['fraction_noun'] = train.apply(lambda row: fraction_noun(row), axis =1)

# train['fraction_adj'] = train.apply(lambda row: fraction_adj(row), axis =1)

# train['fraction_verbs'] = train.apply(lambda row: fraction_verbs(row), axis =1)



# test['fraction_noun'] = test.apply(lambda row: fraction_noun(row), axis =1)

# test['fraction_adj'] = test.apply(lambda row: fraction_adj(row), axis =1)

# test['fraction_verbs'] = test.apply(lambda row: fraction_verbs(row), axis =1)
## Number of words in the text ##

train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))

test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))



## Number of punctuations in the text ##

train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test["mean_word_len"] = test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# clean text

train['text'] = train['text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = False))

test['text'] = test['text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = False))
test['author'] = np.nan

alldata = pd.concat([train, test]).reset_index(drop=True)
#tfidfvec = TfidfVectorizer(analyzer='word', ngram_range = (1,1))

tfidfvec = CountVectorizer(analyzer='word', ngram_range = (1,1),min_df = 1, max_features= 5000)

tfidfdata = tfidfvec.fit_transform(alldata['text'])
tfidfdata.shape
# create dataframe for features

tfidf_df = pd.DataFrame(tfidfdata.todense())
tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]
tfid_df_train = tfidf_df[:len(train)]

tfid_df_test = tfidf_df[len(train):]
# merge into a new data frame with features

train_feats2 = pd.concat([tfid_df_train], axis=1)

test_feats2 = pd.concat([tfid_df_test], axis=1)
# lgb

import lightgbm as lgb

# default parameters

params = {'objective':'multi:softprob',

          'gamma':1,

           'eval_metric':'mlogloss',

          'max_depth': 13,

          'seed':2017,

          'num_class':3,

          'subsample':0.5,

          'eta':0.5

         }
X_train, X_valid, y_train, y_valid = train_test_split(train_feats2, target, train_size = 0.7, stratify = target, random_state = 2017)
dtrain = xgb.DMatrix(data=X_train, label=y_train)

dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

dtest = xgb.DMatrix(data=test_feats2)

watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
model = xgb.train(params, dtrain, 1000, watchlist, maximize=False, verbose_eval=20, early_stopping_rounds=40)
# from sklearn.linear_model import LogisticRegression,SGDClassifier

# from sklearn.ensemble import VotingClassifier

# from sklearn.naive_bayes import MultinomialNB,BernoulliNB, GaussianNB

# from sklearn.svm import SVC

# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import ExtraTreesClassifier



# # clf1 = LogisticRegression(penalty='l1', dual=False, tol=0.0005, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=1)

# # #clf2 = LogisticRegression(penalty='l2', dual=False, tol=0.0005, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=2)

# # #clf3 = LogisticRegression(penalty='l2', dual=False, tol=0.0005, C=1, fit_intercept=True, intercept_scaling=0.2, class_weight=None, random_state=25)

# # #clf1 = BernoulliNB()

# # #clf2 =  GaussianNB()

# # clf3 = MultinomialNB()

# # model = VotingClassifier(estimators=[('lr', clf1), ('svc', clf3)],weights=[3,3], voting='soft')

# model = LogisticRegression(penalty='l1', dual=False, tol=0.0005, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=2)



# # MultinomialNB - term counts is giving higher CV score

# from sklearn.model_selection import cross_val_score

# from sklearn.metrics import accuracy_score, make_scorer, log_loss

# print(cross_val_score(model, train_feats2, target, cv=5, scoring=make_scorer(accuracy_score)))
# model.fit(train_feats2, target)
preds = model.predict(dtest)
result = pd.DataFrame()

result['id'] = test_id

result['EAP'] = [x[0] for x in preds]

result['HPL'] = [x[1] for x in preds]

result['MWS'] = [x[2] for x in preds]



result.to_csv("result.csv", index=False)