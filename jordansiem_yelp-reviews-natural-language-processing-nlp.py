import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk.data



#Dataset from Data Science bootcamp - https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/

yelp = pd.read_csv('../input/yelp-reviews-data-science-bootcamp/yelp.csv')



yelp['text length'] = yelp['text'].apply(len)
yelp.head()

#Yelp Reviews listing number of star ratings given and descriptors for the post (cool,useful,funny)
yelp.info()
yelp.describe()
yelp['text length'] = yelp['text'].apply(len)
#Length of the rating
yelp['text length']
sns.set_style('white')
#Review length based on number of stars given
g = sns.FacetGrid(yelp,col='stars')

g.map(plt.hist,'text length',bins=50)
sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
#dots are outliers
sns.countplot(x='stars',data=yelp,palette='rainbow')
stars = yelp.groupby('stars').mean()
#Number of stars given grouped together and calculated mean
stars
stars.corr()
#Correlations between descriptors and the length of the review
sns.heatmap(stars.corr(),cmap='YlGnBu',annot=True)
yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]
yelp_class
X = yelp_class['text']

y = yelp_class['stars']
#Use bag-of-words model in order to get text into useable format for model
#Count Vectorizer - convert text to matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(X)
from sklearn.model_selection import train_test_split
#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,

test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train,y_train)
#Predict labels
predictions = nb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
#Evaluation of the model with confusion matrix
print(classification_report(y_test,predictions))
#Second check of model with classification report
#Easier way to do vectorization below
from sklearn.feature_extraction.text import TfidfTransformer
#Importing text processor
from sklearn.pipeline import Pipeline
#Pipeline takes care of a lot of the steps above for us.
pipe = Pipeline([('bow',CountVectorizer()), #strings to integer token counts

                 ('tfidf',TfidfTransformer()), #integer counts to weighted TF-IDF scores

                 ('model',MultinomialNB())]) # train on TF-IDF vectors w/ Naive Bayes Classifier
X = yelp_class['text']

y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,

test_size=0.3, random_state=101)
pipe.fit(X_train,y_train)
predictions = pipe.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))