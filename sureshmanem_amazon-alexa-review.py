import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt
print(os.listdir("../input/amazon-alexa-reviews"))
my_alexa = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep = '\t')
my_alexa.head()
my_alexa.keys()
my_alexa['verified_reviews']
my_alexa['variation'].unique()
my_alexa['feedback'].unique() 

# 1 is Positive Feedback and 0 is Negative Feedback
positive_feedback = my_alexa[my_alexa['feedback'] == 1]
positive_feedback.shape # Count of positive feedback
negative_feedback = my_alexa[my_alexa['feedback'] == 0]
negative_feedback.shape # Count of negative feedback
sns.countplot(my_alexa['feedback'],label='count')
sns.countplot(my_alexa['rating'],label='count')
my_alexa['rating'].hist(bins=5)
plt.figure(figsize=(40,15))

sns.barplot(x='variation',y='rating',data=my_alexa, palette='deep')
# 'feedback' is what we are predicting and 'variation' and 'verified_reviews' are used for analysis.

# So, dropping the 'date' and 'rating' fields



my_alexa = my_alexa.drop(['date','rating'],axis=1)
my_alexa.keys() # After 'date' and 'rating' dropped
# Encoding the 'variation' to avoid the dummy trap

variation_dummy = pd.get_dummies(my_alexa['variation'], drop_first = True)
variation_dummy
my_alexa.drop(['variation'],axis=1,inplace=True)
my_alexa.keys()
# Merging the dataframes

my_alexa = pd.concat([my_alexa,variation_dummy],axis = 1)
my_alexa.keys()
# Vectorizing the 'verified_reviews' for analysis

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

alexa_cv = vectorizer.fit_transform(my_alexa['verified_reviews'])
alexa_cv.shape
print(vectorizer.get_feature_names())
print(alexa_cv.toarray())
my_alexa.drop(['verified_reviews'],axis=1,inplace = True)
my_alexa.keys()
encoded_reviews = pd.DataFrame(alexa_cv.toarray())
my_alexa = pd.concat([my_alexa,encoded_reviews],axis = 1)
my_alexa
X = my_alexa.drop(['feedback'],axis=1)
X.shape
y = my_alexa['feedback']
y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# try setting the test_size to 0.3 and 0.4
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# RandomForest Classifer

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

randomforest_classifier = RandomForestClassifier(n_estimators=200,criterion = 'entropy')

randomforest_classifier.fit(X_train,y_train)

# Try the n_estimators with 50,100,150,200,250,300
y_predict_train = randomforest_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train,y_predict_train)
sns.heatmap(cm,annot = True)
print(classification_report(y_train,y_predict_train))
y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_predict))
my_alexa = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', sep = '\t')
my_alexa = pd.concat([my_alexa,pd.DataFrame(alexa_cv.toarray())],axis = 1)
my_alexa.shape
# Adding the length fo the 'verified_review' as a last column in my_alexa dataframe

my_alexa['length'] = my_alexa['verified_reviews'].apply(len)
my_alexa
X = my_alexa.drop(['rating','date','variation','verified_reviews','feedback'],axis=1)
X
y = my_alexa['feedback']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

randomforest_classifier = RandomForestClassifier(n_estimators=300,criterion = 'entropy') # Earlier we had n_estimator as 200

randomforest_classifier.fit(X_train,y_train)



y_predict = randomforest_classifier.predict(X_test)

cm = confusion_matrix(y_test,y_predict)

sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_predict))