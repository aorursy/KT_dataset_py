# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk 
from nltk.corpus import stopwords
#to remove HTML tags from the doc
from bs4 import BeautifulSoup 
#removing numbers,punctuations,i.e regular expressions from the doc
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

import os
print(os.listdir("../input"))

train_data = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train_data.shape
train_data.head(5)
# view the first review
train_data.review[0]
# html tags and comments are removed and stored in sample1

sample1 = BeautifulSoup(train_data.review[0],"html.parser")

# using get_text() we can see only text in html doc

print(sample1.get_text())
# a '^' within square brackets searches anything other than the one on it
# hence here it matches everything from numbers and punctuations etc , leaving only the words

letters_only = re.sub("[^a-zA-Z]"," ",sample1.get_text())
print(letters_only)
# changing all the words to lowercase to create a bag of words later

lower_case = letters_only.lower()

# the whole doc is now split to create an array from which most common words called "stop words" will be removed

words = lower_case.split()
import nltk
nltk.download('stopwords')

# most common stopwords used in english language

print(stopwords.words("english"))
# removing  stopwords from sample1 so that relevant words can be filtered out and stored in words

words = [w for w in words if w not in stopwords.words("english")]
print(words)
# the above code cleans only one review , let's make a function to clean all the reviews
def review_to_words(raw_review):
    #remove html using BeautifulSoup
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()
    #removing raw letters,numbers,punctuations
    letters_only = re.sub("[^a-zA-Z]"," ",review_text)
    #creating an array , resolving whitespaces
    words = letters_only.lower().split()
    #create an array of stopwords so that we don't have to access corpus to search for a stopword
    stop = set(stopwords.words("english"))
    #removing stopwords from the raw_review
    meaningful_words = [w for w in words if w not in stop]
    #return a string with only the words that are important
    return(" ".join(meaningful_words))
# finding the number of reviews
num_rev = train_data.review.size
print(num_rev)
# storing all cleaned reviews in one place

cleaned_rev = []
for i in range(num_rev):
    cleaned_rev.append(review_to_words(train_data.review[i]))
# creating a function, vectorizer to convert the words into vectors

vectorizer = CountVectorizer(analyzer="word",
                            preprocessor=None,
                            stop_words="english",
                            max_features=5000)
# converting reviews from text into features

train_data_features = vectorizer.fit_transform(cleaned_rev)

#change the classifier into array

train_data_features = train_data_features.toarray()
X = train_data_features

#dependent variable,y will be 1 for positive and 0 for negative review

y = train_data.sentiment 
# 25000 rows and 5000 features

print (X.shape) 
print (y.shape) 
# splitting the training data into test and train

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=123)
# Applying MultinomialNaiveBayes for classification 

naive = MultinomialNB()
classifier = naive.fit(X_train,y_train)
predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(predict,y_test)
cm
accuracy = cm.trace()/cm.sum()
print(accuracy)
# loading test data for prediction

test_data = pd.read_csv("../input/testData.tsv",header=0, delimiter="\t", quoting=3)
test_data.head(2)
# preprocessing of test data

number_of_review = len(test_data["review"])
print(number_of_review)

# removing all punctuations,numbers, etc from test data

clean_review =[]
for i in range(number_of_review):
    clean_review.append(review_to_words(test_data["review"][i]))
    
# converting text into features and features to array

test_data_features = vectorizer.fit_transform(clean_review)
test_data_features = test_data_features.toarray()

# predicting test data by the classifier

y_pred_M = classifier.predict(test_data_features)

# accuracy and f1 score

print(accuracy_score(y,y_pred_M))
print(f1_score(y,y_pred_M))
# Applying BernolliNaiveBayes Classifier to training data 

BernNB = BernoulliNB(binarize = 0.01)
BernNB.fit(X_train,y_train)
print(BernNB)

# applying classifier to the test data

y_pred_B = BernNB.predict(test_data_features)
print (accuracy_score(y,y_pred_B))
print (f1_score(y,y_pred_B))

# since accuracy and f1_score are slightly higher in MultinomialNaiveBayes, 
# predicted value of that model is used for the submission.

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column

output = pd.DataFrame( data={"id":test_data["id"], "sentiment": y_pred_M} )

# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )