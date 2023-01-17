# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn
bbc_text = pd.read_csv('../input/bbc-fulltext-and-category/bbc-text.csv')

bbc_text
bbc_text.category.unique()
bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})

bbc_text.category.unique()
bbc_text.info()
bbc_text.shape
# bbc_news = bbc_text.values



X = bbc_text.text

y = bbc_text.category



#split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 1)

print(X_train)

print(y_train)
# countVectorizer



from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words = 'english')
# fit the vectorizer on the training data



vec.fit(X_train)

print(len(vec.get_feature_names()))

vec.vocabulary_
# another way of representing the features

X_transformed = vec.transform(X_train)

X_transformed
print(X_transformed)
X_transformed.toarray()
# convert X_transformed to sparse matrix, just for readability.

pd.DataFrame(X_transformed.toarray(), columns= [vec.get_feature_names()])
# for test data

X_test_transformed = vec.transform(X_test)

X_test_transformed
print(X_test_transformed)
# convert X_transformed to sparse matrix, just for readability

pd.DataFrame(X_test_transformed.toarray(), columns= [vec.get_feature_names()])
from sklearn.linear_model import LogisticRegression



logit = LogisticRegression()

logit.fit(X_transformed, y_train)
# fit

logit.fit(X_transformed,y_train)



# predict class

y_pred_class = logit.predict(X_test_transformed)



# predict probabilities

y_pred_proba = logit.predict_proba(X_test_transformed)
# printing the overall accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
confusion = metrics.confusion_matrix(y_test, y_pred_class)

print(confusion)

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

TP = confusion[1, 1]
sensitivity = TP / float(FN + TP)

print("sensitivity",sensitivity)



specificity = TN / float(TN + FP)

print("specificity",specificity)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class, average = 'micro'))

print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class, average = 'micro'))

print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class, average = 'micro'))
from sklearn.naive_bayes import MultinomialNB



nb = MultinomialNB()

nb.fit(X_transformed, y_train)
# fit

nb.fit(X_transformed,y_train)



# predict class

y_pred_class = nb.predict(X_test_transformed)



# predict probabilities

y_pred_proba = nb.predict_proba(X_test_transformed)
# printing the overall accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
# confusion matrix

metrics.confusion_matrix(y_test, y_pred_class)

# help(metrics.confusion_matrix)
confusion = metrics.confusion_matrix(y_test, y_pred_class)

print(confusion)

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

TP = confusion[1, 1]
sensitivity = TP / float(FN + TP)

print("sensitivity",sensitivity)



specificity = TN / float(TN + FP)

print("specificity",specificity)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class, average = 'micro'))

print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class, average = 'micro'))

print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class, average = 'micro'))
s1 = ['FIR against Delhi Minorities Commission chairman for inflammatory content on social media']

vec1 = vec.transform(s1).toarray()

print('Headline:' ,s1)

print(str(list(nb.predict(vec1))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))
relabel = {'0': 'tech', '1': 'business', '2': 'sport', '3': 'entertainment', '4': 'politics'}
s2 = ['Need to restart economy but with caution: Yogi Adityanath at E-Agenda AajTak']

vec2 = vec.transform(s2).toarray()

print('Headline:' ,s2)

print(str(list(nb.predict(vec2))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))
s3 = ['2 doctors attacked in Andhra Pradesh Vijayawada']

vec3 = vec.transform(s3).toarray()

print('Headline:', s3)

print(str(list(nb.predict(vec3))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))
s4 = ['If I bat for an hour, you’ll see a big one: How Dravid spelt doom for Pak']

vec4 = vec.transform(s4).toarray()

print('Headline:', s4)

print(str(list(nb.predict(vec4))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))