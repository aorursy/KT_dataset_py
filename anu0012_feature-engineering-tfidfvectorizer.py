import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
train = pd.read_csv("../input/hotel-review/train.csv")

test = pd.read_csv("../input/hotel-review/test.csv")
# Target Mapping

mapping_target = {'happy':0, 'not happy':1}

train = train.replace({'Is_Response':mapping_target})



# Browser Mapping

mapping_browser = {'Firefox':0, 'Mozilla':0, 'Mozilla Firefox':0,

                  'Edge': 1, 'Internet Explorer': 1 , 'InternetExplorer': 1, 'IE':1,

                   'Google Chrome':2, 'Chrome':2,

                   'Safari': 3, 'Opera': 4

                  }

train = train.replace({'Browser_Used':mapping_browser})

test = test.replace({'Browser_Used':mapping_browser})

# Device mapping

mapping_device = {'Desktop':0, 'Mobile':1, 'Tablet':2}

train = train.replace({'Device_Used':mapping_device})

test = test.replace({'Device_Used':mapping_device})
test_id = test['User_ID']

target = train['Is_Response']
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

    

    # Replace apostrophes with standard lexicons

    txt = txt.replace("isn't", "is not")

    txt = txt.replace("aren't", "are not")

    txt = txt.replace("ain't", "am not")

    txt = txt.replace("won't", "will not")

    txt = txt.replace("didn't", "did not")

    txt = txt.replace("shan't", "shall not")

    txt = txt.replace("haven't", "have not")

    txt = txt.replace("hadn't", "had not")

    txt = txt.replace("hasn't", "has not")

    txt = txt.replace("don't", "do not")

    txt = txt.replace("wasn't", "was not")

    txt = txt.replace("weren't", "were not")

    txt = txt.replace("doesn't", "does not")

    txt = txt.replace("'s", " is")

    txt = txt.replace("'re", " are")

    txt = txt.replace("'m", " am")

    txt = txt.replace("'d", " would")

    txt = txt.replace("'ll", " will")

    

    # More cleaning

    txt = re.sub(r"review", "", txt)

    txt = re.sub(r"Review", "", txt)

    txt = re.sub(r"TripAdvisor", "", txt)

    txt = re.sub(r"reviews", "", txt)

    txt = re.sub(r"Hotel", "", txt)

    txt = re.sub(r"what's", "", txt)

    txt = re.sub(r"What's", "", txt)

    txt = re.sub(r"\'s", " ", txt)

    txt = txt.replace("pic", "picture")

    txt = re.sub(r"\'ve", " have ", txt)

    txt = re.sub(r"can't", "cannot ", txt)

    txt = re.sub(r"n't", " not ", txt)

    txt = re.sub(r"I'm", "I am", txt)

    txt = re.sub(r" m ", " am ", txt)

    txt = re.sub(r"\'re", " are ", txt)

    txt = re.sub(r"\'d", " would ", txt)

    txt = re.sub(r"\'ll", " will ", txt)

    txt = re.sub(r"60k", " 60000 ", txt)

    txt = re.sub(r" e g ", " eg ", txt)

    txt = re.sub(r" b g ", " bg ", txt)

    txt = re.sub(r"\0s", "0", txt)

    txt = re.sub(r" 9 11 ", "911", txt)

    txt = re.sub(r"e-mail", "email", txt)

    txt = re.sub(r"\s{2,}", " ", txt)

    txt = re.sub(r"quikly", "quickly", txt)

    txt = re.sub(r" usa ", " America ", txt)

    txt = re.sub(r" USA ", " America ", txt)

    txt = re.sub(r" u s ", " America ", txt)

    txt = re.sub(r" uk ", " England ", txt)

    txt = re.sub(r" UK ", " England ", txt)

    txt = re.sub(r"india", "India", txt)

    txt = re.sub(r"switzerland", "Switzerland", txt)

    txt = re.sub(r"china", "China", txt)

    txt = re.sub(r"chinese", "Chinese", txt) 

    txt = re.sub(r"imrovement", "improvement", txt)

    txt = re.sub(r"intially", "initially", txt)

    txt = re.sub(r"quora", "Quora", txt)

    txt = re.sub(r" dms ", "direct messages ", txt)  

    txt = re.sub(r"demonitization", "demonetization", txt) 

    txt = re.sub(r"actived", "active", txt)

    txt = re.sub(r"kms", " kilometers ", txt)

    txt = re.sub(r"KMs", " kilometers ", txt)

    txt = re.sub(r" cs ", " computer science ", txt) 

    txt = re.sub(r" upvotes ", " up votes ", txt)

    txt = re.sub(r" iPhone ", " phone ", txt)

    txt = re.sub(r"\0rs ", " rs ", txt) 

    txt = re.sub(r"calender", "calendar", txt)

    txt = re.sub(r"ios", "operating system", txt)

    txt = re.sub(r"gps", "GPS", txt)

    txt = re.sub(r"gst", "GST", txt)

    txt = re.sub(r"programing", "programming", txt)

    txt = re.sub(r"bestfriend", "best friend", txt)

    txt = re.sub(r"dna", "DNA", txt)

    txt = re.sub(r"III", "3", txt) 

    txt = re.sub(r"the US", "America", txt)

    txt = re.sub(r"Astrology", "astrology", txt)

    txt = re.sub(r"Method", "method", txt)

    txt = re.sub(r"Find", "find", txt) 

    txt = re.sub(r"banglore", "Banglore", txt)

    txt = re.sub(r" J K ", " JK ", txt)



    # Emoji replacement

    txt = re.sub(r':\)',r' Happy ',txt)

    txt = re.sub(r':D',r' Happy ',txt)

    txt = re.sub(r':P',r' Happy ',txt)

    txt = re.sub(r':\(',r' Sad ',txt)

    

    # Remove urls and emails

    txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)

    txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)

    

    # Remove punctuation from text

    txt = ''.join([c for c in text if c not in punctuation])

#     txt = txt.replace(".", " ")

#     txt = txt.replace(":", " ")

#     txt = txt.replace("!", " ")

#     txt = txt.replace("&", " ")

#     txt = txt.replace("#", " ")

    

    # Remove all symbols

    txt = re.sub(r'[^A-Za-z0-9\s]',r' ',txt)

    txt = re.sub(r'\n',r' ',txt)

    

    txt = re.sub(r'[0-9]',r' ',txt)

    

    # Replace words like sooooooo with so

    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))

    

    # Split attached words

    #txt = " ".join(re.findall('[A-Z][^A-Z]*', txt))   

    

    if lowercase:

        txt = " ".join([w.lower() for w in txt.split()])

        

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in stops])

    if stemming:

        st = PorterStemmer()

#         print (len(txt.split()))

#         print (txt)

        txt = " ".join([st.stem(w) for w in txt.split()])

    

    if lemmatization:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])



    return txt
# clean description

train['Description'] = train['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))

test['Description'] = test['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=False, stemming=False, lemmatization = False))
test['Is_Response'] = np.nan

alldata = pd.concat([train, test]).reset_index(drop=True)
tfidfvec = CountVectorizer(analyzer='word', ngram_range = (1,1),max_features=20000)

tfidfdata = tfidfvec.fit_transform(alldata['Description'])
tfidfdata.shape
# create dataframe for features

tfidf_df = pd.DataFrame(tfidfdata.todense())
tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]
tfid_df_train = tfidf_df[:len(train)]

tfid_df_test = tfidf_df[len(train):]
# split the merged data file into train and test respectively

train_feats = alldata[~pd.isnull(alldata.Is_Response)]

test_feats = alldata[pd.isnull(alldata.Is_Response)]
### set target variable



train_feats['Is_Response'] = [1 if x == 'happy' else 0 for x in train_feats['Is_Response']]
# merge into a new data frame with tf-idf features

cols = ['Browser_Used','Device_Used']

train_feats2 = pd.concat([train_feats[cols], tfid_df_train], axis=1)

test_feats2 = pd.concat([test_feats[cols], tfid_df_test], axis=1)
from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import MultinomialNB,BernoulliNB, GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier



clf1 = LogisticRegression(penalty='l1', dual=False, tol=0.0005, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=1)

#clf2 = LogisticRegression(penalty='l2', dual=False, tol=0.0005, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=2)

#clf3 = LogisticRegression(penalty='l2', dual=False, tol=0.0005, C=1, fit_intercept=True, intercept_scaling=0.2, class_weight=None, random_state=25)

#clf1 = BernoulliNB()

#clf2 =  GaussianNB()

clf3 = MultinomialNB()

model = VotingClassifier(estimators=[('lr', clf1), ('svc', clf3)],weights=[3,3], voting='soft')
# Naive Bayes 2 - tfidf is giving higher CV score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, make_scorer

print(cross_val_score(mod1, train_feats2, target, cv=5, scoring=make_scorer(accuracy_score)))
mod1.fit(train_feats2, target)
preds = mod1.predict(test_feats2)
result = pd.DataFrame()

result['User_ID'] = test_id

result['Is_Response'] = preds

mapping = {0:'happy', 1:'not_happy'}

result = result.replace({'Is_Response':mapping})



result.to_csv("lr_predicted_result_1.csv", index=False)