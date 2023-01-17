# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
yelp = pd.read_csv('../input/yelp.csv')
yelp.head()
yelp.describe()
yelp.info()
yelp['text'].apply(len)
yelp['text length'] = yelp['text'].apply(len)
yelp.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
FG = sns.FacetGrid(yelp,col='stars')
FG.map(plt.hist,'text length')
sns.boxplot(x='stars',y='text length',data=yelp,palette='coolwarm')
sns.countplot(x='stars',data=yelp,palette='rainbow')
stars = yelp.groupby('stars').mean()
stars
stars.corr()
sns.heatmap(stars.corr(), cmap= 'coolwarm',annot = True)
yelp_stars = yelp[(yelp.stars==1) | (yelp.stars==5)]
X = yelp_stars['text']

y = yelp_stars['stars']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
X = yelp_stars['text']
y = yelp_stars['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
pipeline.fit(X_train,y_train)
predictions__ = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions__))
print(classification_report(y_test,predictions__))
