# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

import string

import seaborn as sns

import spacy



nlp = spacy.load('en')



%matplotlib inline
# Reading data

tweets_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
# Displaying top 5 rows of the data

tweets_df.head()
print(f'We have {tweets_df.shape[0]} rows of data')
print(f'We have {tweets_df.keyword.nunique()} unique values in keyword and {tweets_df.location.nunique()} unique values in location.')
print(f'There are {tweets_df.target.value_counts()[0]} tweets that are not disaster and {tweets_df.target.value_counts()[1]} tweets that are real disaster')
# Pie chart showing distribution of tweets (disaster and no disaster)

target = ['Disaster', 'No Disater']

colors = ['r', 'g']

plt.pie(tweets_df.target.value_counts(), labels=target, colors=colors, startangle=90, autopct='%.1f%%')

plt.show()
lens = tweets_df.text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist();
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



tweets_df['text'] = tweets_df['text'].apply(remove_URL)
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



tweets_df['text'] = tweets_df['text'].apply(remove_html)
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



tweets_df['text'] = tweets_df['text'].apply(remove_emoji)
tweets_df['text'] = tweets_df['text'].str.lower()
def remove_stop_words_and_punct(text):

    doc = nlp(text)

    return ' '.join([str(token) for token in doc if not token.is_stop and not token.is_punct])



tweets_df['text'] = tweets_df['text'].apply(remove_stop_words_and_punct)
def lammetize(text):

    doc = nlp(text)

    return ' '.join([str(token.lemma_) for token in doc]).replace('-PRON-', 'I')



tweets_df['text'] = tweets_df['text'].apply(lammetize)
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression as LR

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
df = tweets_df[['text', 'target']]
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2)
lr = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),min_df=3,strip_accents='unicode', 

                           use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=None)), ('clf', LR())])



lr = lr.fit(X_train, y_train)



lr_predicted = lr.predict(X_test)



print(f1_score(y_test, lr_predicted))

print(accuracy_score(y_test, lr_predicted))

print(confusion_matrix(y_test, lr_predicted))
rf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),min_df=3,strip_accents='unicode', 

                           use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=None)), ('clf', RandomForestClassifier())])



rf = rf.fit(X_train, y_train)



print(f1_score(y_test, rf.predict(X_test)))

print(accuracy_score(y_test, rf.predict(X_test)))

print(confusion_matrix(y_test, rf.predict(X_test)))
et = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),min_df=3,strip_accents='unicode', 

                           use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=None)), ('clf', ExtraTreesClassifier())])



et = et.fit(X_train, y_train)



print(f1_score(y_test, et.predict(X_test)))

print(accuracy_score(y_test, et.predict(X_test)))

print(confusion_matrix(y_test, et.predict(X_test)))
ada = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),min_df=3,strip_accents='unicode', 

                           use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=None)), ('clf', AdaBoostClassifier())])



ada = ada.fit(X_train, y_train)



print(f1_score(y_test, ada.predict(X_test)))

print(accuracy_score(y_test, ada.predict(X_test)))

print(confusion_matrix(y_test, ada.predict(X_test)))
gb = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),min_df=3,strip_accents='unicode', 

                           use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=None)), ('clf', GradientBoostingClassifier())])



gb = gb.fit(X_train, y_train)



print(f1_score(y_test, gb.predict(X_test)))

print(accuracy_score(y_test, gb.predict(X_test)))

print(confusion_matrix(y_test, gb.predict(X_test)))
pred_df = pd.DataFrame([lr.predict(X_test), rf.predict(X_test), et.predict(X_test), ada.predict(X_test), gb.predict(X_test)]).T

pred_df.columns = ['LR', 'RF', 'ET', 'ADA', 'GB']
import statistics
pred_df['Mode'] = pred_df.apply(statistics.mode, axis=1)
print(f1_score(y_test, pred_df['Mode']))

print(accuracy_score(y_test, pred_df['Mode']))

print(confusion_matrix(y_test, pred_df['Mode']))
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



df_test['text'] = df_test['text'].apply(remove_URL)
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



df_test['text'] = df_test['text'].apply(remove_html)
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



df_test['text'] = df_test['text'].apply(remove_emoji)
df_test['text'] = df_test['text'].str.lower()
def remove_stop_words_and_punct(text):

    doc = nlp(text)

    return ' '.join([str(token) for token in doc if not token.is_stop and not token.is_punct])



df_test['text'] = df_test['text'].apply(remove_stop_words_and_punct)
def lammetize(text):

    doc = nlp(text)

    return ' '.join([str(token.lemma_) for token in doc]).replace('-PRON-', 'I')



df_test['text'] = df_test['text'].apply(lammetize)
dfte = df_test['text']
test_pred_df = pd.DataFrame([lr.predict(dfte), rf.predict(dfte), et.predict(dfte), ada.predict(dfte), gb.predict(dfte)]).T

test_pred_df.columns = ['LR', 'RF', 'ET', 'ADA', 'GB']



import statistics



test_pred_df['Mode'] = test_pred_df.apply(statistics.mode, axis=1)
df_subm = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
df_subm['target'] = test_pred_df['Mode']
df_subm.to_csv('submission.csv', index=False)