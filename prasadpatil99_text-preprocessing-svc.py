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
train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
train = train[['text','target']]

test = test[['id','text']]
from nltk.stem import PorterStemmer

import nltk

from nltk.corpus import stopwords

stopwords = set(stopwords.words("english"))

import re
train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

test['text'] = test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
corpus_train = train['text']

corpus_test = test['text']
def replace(text):

    text = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$'," ")  # remove emailaddress

    text = text.str.replace(r'\W+'," ")     # remove symbols

    text = text.str.replace(r' '," ")       # remove punctuations

    text = text.str.replace('\d+'," ")      # remove numbers

    text = text.str.lower()                 # remove capital letters as they does not make any effect

    return text
corpus_train = replace(corpus_train)

corpus_test = replace(corpus_test)
import nltk

nltk.download('wordnet')

from textblob import Word
freq = pd.Series(' '.join(corpus_train).split()).value_counts()[-19500:]

corpus_train = corpus_train.apply(lambda x: " ".join(x for x in x.split() if x not in freq))
freq.head()
freq = pd.Series(' '.join(corpus_test).split()).value_counts()[-10000:]

corpus_test = corpus_test.apply(lambda x: " ".join(x for x in x.split() if x not in freq))
from wordcloud import WordCloud 

import matplotlib.pyplot as plt

def wordcloud(text):

    wordcloud = WordCloud(

        background_color='white',

        max_words=500,

        max_font_size=30, 

        scale=3,

        random_state=5

    ).generate(str(corpus_train))

    fig = plt.figure(figsize=(15, 12))

    plt.axis('off')

    plt.imshow(wordcloud)

    plt.show()

    

wordcloud(corpus_train)
import seaborn as sns

target = train['target']

sns.countplot(target)
from sklearn.feature_extraction.text import TfidfVectorizer



Tfidf_vect = TfidfVectorizer(max_features = 7000)

Tfidf_vect.fit(corpus_train)

X_train = Tfidf_vect.transform(corpus_train)

X_test = Tfidf_vect.transform(corpus_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

parameters = { 

    'gamma': [0.001, 0.01, 0.1, 0.4, 0.6, 0.7, 'auto'], # for complex decision boundary (mainly used for rbf kerel)

    

    'kernel': ['rbf','linear'], # used for different type of data

                                # linear - when data is easy to classify 

                                # rbf - when data is too complex

    

    'C': [0.001, 0.01, 0.1, 1, 1.5, 2, 3, 10], # inverse weight on regularization parameter 

                                               # (how finely to classify, decreasing will prevent overfititing and vice versa)

}

model = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1).fit(X_train, target)

model.cv_results_['params'][model.best_index_]

y_val_pred = model.predict(X_test)
from sklearn.svm import SVC

SVM = SVC(C=1.0, kernel='linear', gamma='auto')

SVM.fit(X_train,target)

SVM_predictions = SVM.predict(X_test)
file_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

file_submission.target = SVM_predictions

file_submission.to_csv("submission.csv", index=False)