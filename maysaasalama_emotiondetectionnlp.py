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
import pandas as pd

text_emotion = pd.read_csv("../input/text_emotion.csv")
text_emotion.head(10)
text_emotion = text_emotion.drop('author', axis=1)
text_emotion.head(5)
text_emotion = text_emotion.drop(text_emotion[text_emotion.sentiment == 'anger'].index)

text_emotion = text_emotion.drop(text_emotion[text_emotion.sentiment == 'boredom'].index)

text_emotion = text_emotion.drop(text_emotion[text_emotion.sentiment == 'enthusiasm'].index)

text_emotion = text_emotion.drop(text_emotion[text_emotion.sentiment == 'empty'].index)

text_emotion = text_emotion.drop(text_emotion[text_emotion.sentiment == 'worry'].index)

text_emotion = text_emotion.drop(text_emotion[text_emotion.sentiment == 'fun'].index)

text_emotion = text_emotion.drop(text_emotion[text_emotion.sentiment == 'relief'].index)
#Making all letters lowercase

text_emotion['content'] = text_emotion['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
text_emotion.head
#Removing Punctuation, Symbols

text_emotion['content'] = text_emotion['content'].str.replace('[^\w\s]',' ')
#Removing Stop Words using NLTK

from nltk.corpus import stopwords

stop = stopwords.words('english')

text_emotion['content'] = text_emotion['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
text_emotion.head(5)
from textblob import Word

text_emotion['content'] = text_emotion['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
import re

def de_repeat(text):

    pattern = re.compile(r"(.)\1{2,}")

    return pattern.sub(r"\1\1", text)
text_emotion['content'] = text_emotion['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
# Code to find the top 10,000 rarest words appearing in the data

freq = pd.Series(' '.join(text_emotion['content']).split()).value_counts()[-10000:]
# Removing all those rarely appearing words from the data

freq = list(freq.index)

text_emotion['content'] = text_emotion['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#Encoding output labels 'happiness' as '0' , 'hate' as '1' , 'love' as '2' , 'neutral' as '3' , 'sadness' as '4' , 'surprise' as '5' ,  'worry' as '6' 

from sklearn import preprocessing

lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(text_emotion.sentiment.values)
# Splitting into training and testing data in 90:10 ratio

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(text_emotion.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
# Extracting TF-IDF parameters

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,5))
X_train_tfidf = tfidf.fit_transform(X_train)

X_val_tfidf = tfidf.fit_transform(X_val)
# Extracting Count Vectors Parameters

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word')

count_vect.fit(text_emotion['content'])
X_train_count =  count_vect.transform(X_train)

X_val_count =  count_vect.transform(X_val)
from sklearn.metrics import accuracy_score
# Model 1: Multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_tfidf, y_train)

y_pred = nb.predict(X_val_tfidf)

print('naive bayes TF-IDF accuracy %s' % accuracy_score(y_pred, y_val))
# Model 2: Linear SVM

from sklearn.linear_model import SGDClassifier

lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)

lsvm.fit(X_train_tfidf, y_train)

y_pred = lsvm.predict(X_val_tfidf)

print('svm using tfidf accuracy %s' % accuracy_score(y_pred, y_val))
# Model 3: logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1)

logreg.fit(X_train_tfidf, y_train)

y_pred = logreg.predict(X_val_tfidf)

print('log reg tfidf accuracy %s' % accuracy_score(y_pred, y_val))
# Model 4: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500)

rf.fit(X_train_tfidf, y_train)

y_pred = rf.predict(X_val_tfidf)

print('random forest tfidf accuracy %s' % accuracy_score(y_pred, y_val))
# Model 1: Multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_count, y_train)

y_pred = nb.predict(X_val_count)

print('naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_val))
# Model 2: Linear SVM

from sklearn.linear_model import SGDClassifier

lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)

lsvm.fit(X_train_count, y_train)

y_pred = lsvm.predict(X_val_count)

print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_val))
# Model 3: Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1)

logreg.fit(X_train_count, y_train)

y_pred = logreg.predict(X_val_count)

print('log reg count vectors accuracy %s' % accuracy_score(y_pred, y_val))
# Model 4: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500)

rf.fit(X_train_count, y_train)

y_pred = rf.predict(X_val_count)

print('random forest with count vectors accuracy %s' % accuracy_score(y_pred, y_val))
tweets = pd.DataFrame(['I am very happy today! The atmosphere looks cheerful',

                       'His death broke my heart. It was a sad day',

                      'I am very happy today!',

                      'cant fall asleep',

                      'Happy Mothers Day All my love',

                      'please to meet you',

                      'she is crying'])
tweets[0] = tweets[0].str.replace('[^\w\s]',' ')

from nltk.corpus import stopwords

stop = stopwords.words('english')

tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
from textblob import Word

tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# Extracting Count Vectors feature from our tweets

tweet_count = count_vect.transform(tweets[0])
tweet_pred = logreg.predict(tweet_count)

print(tweet_pred)