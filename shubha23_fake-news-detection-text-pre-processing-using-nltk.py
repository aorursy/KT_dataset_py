# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

import string as st

import re

import nltk

from nltk import PorterStemmer, WordNetLemmatizer

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/textdb3/fake_or_real_news.csv')

data.shape
data.head()
# Check how the labels are distributed

print(np.unique(data['label']))

print(np.unique(data['label'].value_counts()))
# Remove all punctuations from the text



def remove_punct(text):

    return ("".join([ch for ch in text if ch not in st.punctuation]))
data['removed_punc'] = data['text'].apply(lambda x: remove_punct(x))

data.head()
''' Convert text to lower case tokens. Here, split() is applied on white-spaces. But, it could be applied

    on special characters, tabs or any other string based on which text is to be seperated into tokens.

'''

def tokenize(text):

    text = re.split('\s+' ,text)

    return [x.lower() for x in text]
data['tokens'] = data['removed_punc'].apply(lambda msg : tokenize(msg))

data.head()
# Remove tokens of length less than 3

def remove_small_words(text):

    return [x for x in text if len(x) > 3 ]
data['filtered_tokens'] = data['tokens'].apply(lambda x : remove_small_words(x))

data.head()
''' Remove stopwords. Here, NLTK corpus list is used for a match. However, a customized user-defined 

    list could be created and used to limit the matches in input text. 

'''

def remove_stopwords(text):

    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]
data['clean_tokens'] = data['filtered_tokens'].apply(lambda x : remove_stopwords(x))

data.head()
# Apply lemmatization on tokens

def lemmatize(text):

    word_net = WordNetLemmatizer()

    return [word_net.lemmatize(word) for word in text]
data['lemma_words'] = data['clean_tokens'].apply(lambda x : lemmatize(x))

data.head()
# Create sentences to get clean text as input for vectors



def return_sentences(tokens):

    return " ".join([word for word in tokens])
data['clean_text'] = data['lemma_words'].apply(lambda x : return_sentences(x))

data.head()
# Generate a basic word cloud 

from wordcloud import WordCloud, ImageColorGenerator



text = " ".join([x for x in data['clean_text']])

# Create and generate a word cloud image:

wordcloud = WordCloud(max_font_size=30, max_words=1000).generate(text)



# Display the generated image:

plt.figure(figsize= [20,10])

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# Prepare data for the model. Convert label in to binary



data['label'] = [1 if x == 'FAKE' else 0 for x in data['label']]

data.head()
# Split the dataset



X_train,X_test,y_train,y_test = train_test_split(data['clean_text'], data['label'], test_size=0.2, random_state = 5)



print(X_train.shape)

print(X_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer()

tfidf_train = tfidf.fit_transform(X_train)

tfidf_test = tfidf.transform(X_test)



print(tfidf_train.toarray())

print(tfidf_train.shape)

print(tfidf_test.toarray())

print(tfidf_test.shape)
# Passive Aggresive Classifier

pac = PassiveAggressiveClassifier(max_iter=50)

pac.fit(tfidf_train,y_train)



pred = pac.predict(tfidf_test)

print("Accuracy score : {}".format(accuracy_score(y_test, pred)))

print("Confusion matrix : \n {}".format(confusion_matrix(y_test, pred)))
# Logistic Regression model

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(max_iter = 500)

lr.fit(tfidf_train, y_train)

print('Logistic Regression model fitted..')



pred = lr.predict(tfidf_test)

print("Accuracy score : {}".format(accuracy_score(y_test, pred)))

print("Confusion matrix : \n {}".format(confusion_matrix(y_test, pred)))
import xgboost

from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(tfidf_train, y_train)



print('XGBoost Classifier model fitted..')

pred = xgb.predict(tfidf_test)

print("Accuracy score : {}".format(accuracy_score(y_test, pred)))

print("Confusion matrix : \n {}".format(confusion_matrix(y_test, pred)))
import lightgbm

from lightgbm import LGBMClassifier



lgbm = LGBMClassifier()

lgbm.fit(tfidf_train, y_train)



print('LightGBM Classifier model fitted..')

pred = lgbm.predict(tfidf_test)

print("Accuracy score : {}".format(accuracy_score(y_test, pred)))

print("Confusion matrix : \n {}".format(confusion_matrix(y_test, pred)))