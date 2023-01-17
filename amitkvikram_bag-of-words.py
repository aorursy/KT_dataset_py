# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/labeledTrainData.tsv', header = 0, 
                    delimiter= "\t", quoting = 3)
print(train.shape)
display(train.head())

example1 = BeautifulSoup(train['review'][0])
print(train['review'][0])
print("\n\n")
print(example1.get_text())
import re
letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text())
print(letters_only)
lower_case = letters_only.lower()
words = lower_case.split()
print(words)
import nltk
from nltk.corpus import stopwords

words = [w for w in words if not w in stopwords.words("english")]
print(words)
stops = set(stopwords.words("english"))
def review_to_words(review):
    #1. Remove HTML TAGS
    review = BeautifulSoup(review).get_text()
    #2. Remove punctuation and numbers
    letters_only = re.sub("[^a-zA-Z]", " ", review)
    #3. Convert to lower case and split into words
    lower_case = letters_only.lower()
    words = lower_case.split()
    #4. Remove Stop Words
    words = [w for w in words if not w in stops]
    #5. Return string
    return " ".join(words)
review_to_words(train['review'][0])
num_reviews = train.shape[0]
train_corpus = []
for i in tqdm_notebook(range(num_reviews)):
    train_corpus.append(review_to_words(train['review'][i]))
len(train_corpus[-1].split())
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = 'word', tokenizer = None, 
                            preprocessor = None, stop_words = None, 
                            max_features = 8000)
vectorizer.fit(train_corpus)
X_train = vectorizer.transform(train_corpus).toarray()
y_train = train.sentiment.values.reshape(-1,1)
print(X_train.shape, y_train.shape)
print(len(train_corpus[0]))
print(np.sum(X_train, axis = 1))
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train , y_train.reshape(-1, ), 
                                                test_size = 0.2,
                                                stratify = y_train.reshape(-1, ), 
                                                random_state = 42)
print("Training Data = ", X_train.shape, y_train.shape)
print("Validation Date = ", X_val.shape, y_val.shape)
test = pd.read_csv("../input/testData.tsv", delimiter = "\t", 
                  header = 0, quoting = 3)
print(test.shape)
test.head()
num_test_reviews = test.shape[0]
test_corpus = []
for i in tqdm_notebook(range(num_test_reviews)):
    test_corpus.append(review_to_words(test['review'][i]))
print(len(test))
X_test = vectorizer.transform(test_corpus).toarray()
print(X_test.shape)
X_test
print(X_train)
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix
clfrNB = naive_bayes.MultinomialNB()
clfrNB.fit(X_train, y_train.reshape(-1, ))
y_pred = clfrNB.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred)
y_pred = clfrNB.predict(X_val)
cm_val = confusion_matrix(y_val, y_pred)
print("Training Accuracy = ", clfrNB.score(X_train, y_train))
print("Validation Accuracy", clfrNB.score(X_val, y_val))
print(cm_train)
print(cm_val)
y_test = clfrNB.predict(X_test)
print(y_test.shape)
y_test
sample = pd.read_csv("../input/sampleSubmission.csv", header = 0)
sample.head()
sub = pd.DataFrame()
sub['id'] = sample['id'].values
sub['sentiment'] = y_test
sub.to_csv("naiveBayesPred.csv", index = False)
sub.head()
from sklearn.ensemble import RandomForestClassifier
clfrRMC = RandomForestClassifier(n_estimators= 500, criterion = "gini", 
                                max_features = "auto", max_depth = 8, 
                                min_samples_split = 2, random_state = 42)
clfrRMC.fit(X_train, y_train.reshape(-1, ))
y_pred = clfrRMC.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred)
y_pred = clfrRMC.predict(X_val)
cm_val = confusion_matrix(y_val, y_pred)
print("Training Accuracy = ", clfrRMC.score(X_train, y_train))
print("Validation Accuracy = ", clfrRMC.score(X_val, y_val))
print(cm_train)
print(cm_val)
y_test = clfrRMC.predict(X_test)
print(y_test.shape)
print(y_test)
sub = pd.DataFrame()
sub['id'] = sample['id']
sub['sentiment'] = y_test
sub.to_csv("randomForestPred.csv", index = False)
print(sub.shape)
sub.head()
from sklearn.linear_model import LogisticRegression
clfrLR = LogisticRegression(penalty = 'l2', C = 0.005)
clfrLR.fit(X_train, y_train)
y_pred = clfrLR.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred)
y_pred = clfrLR.predict(X_val)
cm_val = confusion_matrix(y_val, y_pred)
print("Training Accuracy = ", clfrLR.score(X_train, y_train))
print("Validation Accuracy = ", clfrLR.score(X_val, y_val))
print(cm_train)
print(cm_val)
y_test = clfrLR.predict(X_test)
print(y_test.shape)
print(y_test)
sub = pd.DataFrame()
sub['id'] = sample['id']
sub['sentiment'] = y_test
sub.to_csv("LogisticPrediction.csv", index = False)
print(sub.shape)
sub.head()




