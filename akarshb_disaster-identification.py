import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

import seaborn as sns

import missingno as msno

import nltk

import string

from nltk.tokenize import word_tokenize

import spacy

from collections import Counter

from spacy.lang.en import English

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

spacy.load('en')

puncts = string.punctuation

from spacy.lang.en.stop_words import STOP_WORDS

nlp = English()
data = pd.read_csv('../input/nlp-getting-started/train.csv')
data.head()
data.info()
#Location Nulls

7613-5080
#Keyword Nulls

7613-7552
msno.heatmap(data)
msno.dendrogram(data)
data.fillna(value=' ',axis=1,inplace=True)
data.head()
data.info()
data['combined'] = data.keyword + ' ' + data.location + ' ' + data.text

data.head()
ax =sns.countplot(x='target',data=data)
def unique_list(l):

    ulist = []

    [ulist.append(x) for x in l if x not in ulist]

    return ulist
def clean_text(docs):

    texts = []

    for doc in docs:

        doc = ' '.join(unique_list(doc.split()))

        doc = nlp(doc, disable=['parser', 'ner'])

        tokens = [token.lemma_.lower().strip() for token in doc if token.lemma_ != '-PRON-']

        tokens = [token for token in tokens if token not in STOP_WORDS]

        tokens = ' '.join(tokens)

        texts.append(tokens)

    return pd.Series(texts)
combined_text = [text for text in data.combined.str.strip()]

combined_text_clean = clean_text(combined_text)

data['combined_clean'] = combined_text_clean
def text_preprocess(tweet):

    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    tweet = re.sub(r'#|@|\.|amp', '', tweet)

    tweet = re.sub(r'['+puncts+']+', ' ', tweet)

    tweet = re.sub(r'\d+','',tweet)

    return str(tweet).strip()

data.combined_clean = data.combined_clean.apply(text_preprocess)
disaster_text= [text for text in data[data.target == 1].combined_clean.str.strip()]

disaster_text = ' '.join(disaster_text).split()

non_disaster_text= [text for text in data[data.target == 0].combined_clean.str.strip()]

non_disaster_text = ' '.join(non_disaster_text).split()

disaster_text_count = Counter(disaster_text)

non_disaster_text_count = Counter(non_disaster_text)
disaster_common_words = [word[0] for word in disaster_text_count.most_common(25)]

disaster_common_counts = [word[1] for word in disaster_text_count.most_common(25)]

figure = plt.figure(figsize = (20,5))

ax = sns.barplot(x=disaster_common_words,y=disaster_common_counts)
non_disaster_common_words = [word[0] for word in non_disaster_text_count.most_common(25)]

non_disaster_common_counts = [word[1] for word in non_disaster_text_count.most_common(25)]

figure = plt.figure(figsize = (20,5))

ax = sns.barplot(x=non_disaster_common_words,y=non_disaster_common_counts)
tfidf_vector = TfidfVectorizer()

text_tfidf = tfidf_vector.fit_transform(data.combined_clean)
X_train, X_test, y_train, y_test = train_test_split(text_tfidf, data.target, test_size=0.3, random_state=42)
log_clf = LogisticRegression(random_state=42).fit(X_train, y_train)

predictions = log_clf.predict(X_test)

print(classification_report(y_test, predictions))
clf = DecisionTreeClassifier(random_state=42)

clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(classification_report(y_test, predictions))
clf = RandomForestClassifier(random_state=42,n_estimators=10)

clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(classification_report(y_test, predictions))
parameters = {'loss': ['hinge', 'squared_hinge','log'], 

'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000],

'penalty':['l2', 'l1','elasticnet']}

model = SGDClassifier(max_iter=1000,random_state=42)

clf= GridSearchCV(model,parameters,cv=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, predictions))
test_data = pd.read_csv('../input/nlp-getting-started/test.csv')
test_data.head()
test_data.fillna(value=' ',axis=1,inplace=True)

test_data['combined'] = test_data.keyword + ' ' + test_data.location + ' ' + test_data.text

combined_text = [text for text in test_data.combined.str.strip()]

combined_text_clean = clean_text(combined_text)

test_data['combined_clean'] = combined_text_clean

test_data.combined_clean = test_data.combined_clean.apply(text_preprocess)
test_data.head()
test_text_tfidf = tfidf_vector.transform(test_data.combined_clean)
log_predictions = log_clf.predict(test_text_tfidf)

log_predictions