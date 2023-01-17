import pandas as pd

import numpy as np

import matplotlib.pyplot

%matplotlib inline

import seaborn as sns

import nltk



import string

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report
train = pd.read_csv("../input/train.csv")
train.describe()
train.head()
train.head()
train['keyword'].unique()
train['location'].unique()
train.describe()
train['length'] = train['text'].apply(len)
train[train['keyword'] == 'ablaze']
sns.distplot(train[train['target'] == 0]['length'])
sns.distplot(train[train['target'] == 1]['length'])
def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]



train['text'].head(5).apply(text_process)
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
X= train['text']

y= train['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))