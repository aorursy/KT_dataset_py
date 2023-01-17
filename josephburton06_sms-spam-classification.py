import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn import tree



import seaborn as sns



import unicodedata

import re

import json



import nltk

from nltk.tokenize.toktok import ToktokTokenizer

from nltk.corpus import stopwords



import sms_helpers

from sms_helpers import original_word_count

from sms_helpers import basic_clean

from sms_helpers import article_word_count

from sms_helpers import article_percent

from sms_helpers import text_prep

from sms_helpers import remove_stopwords



import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_table('SMSSpamCollection.txt', header=None)

df.columns = ['result', 'original']
df.shape
df = text_prep(df)

df.head(3)
df[df.article_cnt != df.clean_cnt]
df.groupby('result')[['article_per_kept']].agg(['mean', 'min', 'max'])
df[df.article_per_kept > 3]
df[df.article_per_kept < .8]
df.groupby('result').count()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score



from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df.clean)

y = df.result



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2)



train = pd.DataFrame(dict(actual=y_train))

test = pd.DataFrame(dict(actual=y_test))



lm = LogisticRegression().fit(X_train, y_train)



train['predicted'] = lm.predict(X_train)



print('Accuracy: {:.2%}'.format(accuracy_score(train.actual, train.predicted)))

print('---')

print('Confusion Matrix')

print(pd.crosstab(train.predicted, train.actual))

print('---')

print(classification_report(train.actual, train.predicted))
test['predicted'] = lm.predict(X_test)



print('Accuracy: {:.2%}'.format(accuracy_score(test.actual, test.predicted)))

print('---')

print('Confusion Matrix')

print(pd.crosstab(test.predicted, test.actual))

print('---')

print(classification_report(test.actual, test.predicted))