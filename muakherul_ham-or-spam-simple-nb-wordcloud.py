# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
# Any results you write to the current directory are saved as output.
df = pd .read_csv('../input/spam.csv', encoding='ISO-8859-1')
df.head()
df = df[['v1', 'v2']]
df.describe()
df['res'] = df.v1.map({'ham':0, 'spam':1})
df = df[['v1', 'v2', 'res']]
df.head()
df.groupby(['v1'])['res'].count().plot(kind='bar', fontsize='13')
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
import re
def tokenz(x):
    spams = ' '.join(x).lower()
    spams = re.sub('[^a-z]+', ' ', spams)
    spams = nltk.word_tokenize(spams)
    spams = [i for i in spams if len(i) > 1]

    stop_words = list(get_stop_words('en'))
    spams_tokenize = [word for word in spams if word not in stop_words]
    spams_tokenize = ' '.join(spams_tokenize)
    return spams_tokenize

spam_words = tokenz(df[df.v1 == 'spam'].v2)
ham_words = tokenz(df[df.v1 == 'ham'].v2)
wc = WordCloud(width=600,height=300)

cld = wc.generate(spam_words)
plt.figure(figsize=(10,5), dpi=200, facecolor='k')
plt.imshow(cld)
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('WordCloud for Spam message')
plt.show()

cld = wc.generate(ham_words)

plt.figure(figsize=(8,4), dpi=200, facecolor='k')
plt.imshow(cld)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
X = df.v2
y = df.res
print(X.shape)
print(y.shape)
#Train and split data to train and test
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words='english')
#vec.fit(X_train)
#X_train_dtm = vec.transform(X_train)
X_train_dtm = vec.fit_transform(X_train)
X_train_dtm.shape
X_test_dtm = vec.transform(X_test)
X_test_dtm.shape
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
pred = nb.predict(X_test_dtm)
metrics.accuracy_score(pred,y_test)
metrics.confusion_matrix(y_test, pred)
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

parameters = {'alpha':(0,1), 'fit_prior': (True, False)}

gridSearch = GridSearchCV(MultinomialNB(), parameters, scoring='accuracy')
gridSearch.fit(X_train_dtm, y_train)

print(gridSearch.best_score_)
print(gridSearch.best_params_)




