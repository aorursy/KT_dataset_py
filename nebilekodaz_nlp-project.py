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
import nltk as nl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
text=pd.read_csv('../input/yelp.csv')
text.head()
text.info()
text.describe()
text['length'] =text['text'].apply(len)
text.head()
g = sns.FacetGrid(text,col='stars')
g.map(plt.hist,'length')
plt.figure(figsize=(15,8))
sns.boxplot(x="stars", y="length", data=text,palette='rainbow')
sns.countplot(x='stars',data=text)
stars=text.groupby(['stars']).mean()
stars
corrstars=stars.corr()
corrstars
sns.heatmap(corrstars,cmap='coolwarm',annot=True)
yelp_class=text[(text['stars']==1) | (text['stars']==5)]
len(yelp_class)
X=yelp_class['text']
y=yelp_class['stars']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb_fit=nb.fit(X_train,y_train)
all_predictions = nb.predict(X_test)
print(all_predictions)
from sklearn.metrics import classification_report
print (classification_report(y_test, all_predictions))
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
yelp_class=text[(text['stars']==1) | (text['stars']==5)]
len(yelp_class)
X=yelp_class['text']
y=yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(classification_report(predictions,y_test))
