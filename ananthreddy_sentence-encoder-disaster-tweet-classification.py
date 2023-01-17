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
import sys

import matplotlib.pyplot as plt

from tqdm import tqdm, trange



%matplotlib inline
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print (train_data.shape)

train_data.head()
train_data['target'].value_counts()
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print (test_data.shape)

test_data.head()
import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()



def clean_sentence(sentence):

    sent_tokens = sentence.split(' ')

    sent_tokens = [w for w in sent_tokens if not w in stop_words]

    sent_tokens = [wordnet_lemmatizer.lemmatize(w) for w in sent_tokens]

    clean_sentence = ','.join(sent_tokens)

    clean_sentence = clean_sentence.replace(',', ' ')

    return (clean_sentence)



tqdm.pandas()

train_data['clean_text'] = train_data['text'].progress_apply(lambda x:clean_sentence(x))

test_data['clean_text'] = test_data['text'].progress_apply(lambda x:clean_sentence(x))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train_data['clean_text'], train_data['target'], test_size = 0.25, random_state = 12)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()



X_train_vect = vectorizer.fit_transform(X_train)

X_test_vect = vectorizer.transform(X_test)
test_vect = vectorizer.transform(test_data['clean_text'])
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV



params = {'boosting_type': ['gbdt'],

         'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.20],

         'num_leaves': [10, 31, 40, 100],

         'n_jobs': [-1]}

lgb_classifier = lgb.LGBMClassifier()



gs_classifier = GridSearchCV(

    estimator = lgb.LGBMClassifier(),

    param_grid = params

)
gs_classifier.fit(X_train_vect, y_train)

print (gs_classifier.best_score_)

print (gs_classifier.best_estimator_)
y_pred = gs_classifier.predict(X_test_vect)
from sklearn.metrics import classification_report



print (classification_report(y_pred, y_test))
test_prediction = gs_classifier.predict(test_vect)
submission = pd.DataFrame({'id': test_data['id'], 'target': test_prediction})

submission.to_csv('submission.csv', index = False)