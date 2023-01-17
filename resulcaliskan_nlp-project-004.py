import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

from pprint import pprint

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = []

with open('../input/tail_review_100000.json') as f:

    for line in f:

        data.append(json.loads(line))

pprint(data[0])
yelp = pd.DataFrame(data,columns= ["business_id", "date", "review_id", "stars", "text", "type", "user_id", "cool", "useful", "funny"])

yelp.head()
yelp.info()
yelp.describe()
# add a new column.

yelp['text length']=yelp["text"].apply(len)
yelp.tail()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

%matplotlib inline
# Using FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings.

g = sns.FacetGrid(yelp,col='stars')

g.map(plt.hist,'text length')
# Creating a boxplot of text length for each star category.

sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
# Creatinga countplot of the number of occurrences for each type of star rating.

sns.countplot(x='stars', data=yelp, palette='rainbow')
# groupby to get the mean values of the numerical columns



stars = yelp.groupby('stars').mean().drop("type", axis=1)

stars
# corr() method on that groupby dataframe to produce this dataframe



stars.corr()
# heatmap based off that .corr() dataframe



sns.heatmap(stars.corr(), cmap="coolwarm", annot=True)
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

X = yelp_class['text']

y = yelp_class['stars']
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

print("\n")

print(classification_report(y_test,predictions))
from sklearn.feature_extraction.text import  TfidfTransformer

from sklearn.pipeline import Pipeline
# Now create a pipeline with the CountVectorizer(), TfidfTransformer(),MultinomialNB()

pipeline = Pipeline([

    ('bow', CountVectorizer()),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
X = yelp_class['text']

y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))