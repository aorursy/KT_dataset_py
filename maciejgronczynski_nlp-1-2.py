# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib

import matplotlib.pyplot as plt



#SKLEARN

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

#OTHER LIBRARIES

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import string

import re

import nltk











# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print('Loading Successful.')
test_data  = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print('Loading Successful.')
train_data.head()
train_data.info()
train_data_no_duplicates = train_data.drop_duplicates(subset=['text'])
x1 = len(train_data) #=7613

x2 = len(set(train_data['text'])) #=7498

number_of_duplicated_records = (x1 - x2) 

print('Number of duplicated records is:',number_of_duplicated_records)
train_data_no_duplicates.head()
train_data_no_duplicates.shape
x1 = len(train_data_no_duplicates) #=7613

x2 = len(set(train_data_no_duplicates['text'])) #=7498

number_of_duplicated_records = (x1 - x2) 

print('Number of duplicated records is:',number_of_duplicated_records)
def remove_punct(text):

    text_nopunct = ''.join([char for char in text if char not in string.punctuation])

    return text_nopunct



train_data_no_duplicates['text'] = train_data_no_duplicates['text'].apply(lambda x: remove_punct(x))

#this is how you can drop any column. --> train_data.drop('text_clean', axis=1, inplace=True)

train_data_no_duplicates.head()
#train_data_no_duplicates=train_data_no_duplicates.drop('location',1)



#test_data=test_data.drop('location',1)
train_data_no_duplicates.head()



vectorizer = CountVectorizer(analyzer='word', binary=True)

vectorizer.fit(train_data_no_duplicates['text'])

X = vectorizer.transform(train_data_no_duplicates['text']).todense()

y = train_data_no_duplicates['target'].values

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)





model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



f1score = f1_score(y_test, y_pred)

print(f"Model Score: {f1score * 100} %")
tweets_test = test_data['text']

test_X = vectorizer.transform(tweets_test).todense()

test_X.shape
lr_pred = model.predict(test_X)
sub['target'] = lr_pred

sub.to_csv("submission.csv", index=False)

sub.head()