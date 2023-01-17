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
dataset = pd.read_csv("../input/spam.csv",encoding='latin-1')
dataset.head()
dataset.describe()
# the columns unnamed2 , unnamed3 , unnamed4 can be dropped

dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1":"class", "v2":"text"})

dataset.head()
dataset.groupby('class').describe()
X = dataset.iloc[:,1:2].values
X
y = dataset.iloc[:,0:1].values
y
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train
y_train
np.unique(y,return_counts=True)
np.unique(y_train,return_counts=True)
np.unique(y_test,return_counts=True)
# text preprocessing



import re

import nltk

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

X_test[0][0]
   # lets preprocess first row in X_test 

cleanr = re.compile('<.*?>')

review = re.sub(cleanr, '', X_test[0][0])

review = re.sub('[^a-zA-Z]', ' ', review)

review = review.lower()

review = review.split()

ps = PorterStemmer()

review = [ps.stem(word) for word in review  if not word in set(stopwords.words('english'))]

   
review
X_test
corpus_test = []

for i in range(len(X_test)):

    cleanr = re.compile('<.*?>')

    review = re.sub(cleanr, '', X_test[i][0])

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review  if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus_test.append(review)
corpus_train = []

for i in range(len(X_train)):

    cleanr = re.compile('<.*?>')

    review = re.sub(cleanr, '', X_train[i][0])

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review  if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus_train.append(review)
corpus_train
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))

X_train_after = tfidf.fit_transform(corpus_train).toarray()
X_train_after
X_test_after = tfidf.transform(corpus_test).toarray()
X_test_after
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

y_train = labelencoder_y.fit_transform(y_train.ravel())
y_test = labelencoder_y.fit_transform(y_test.ravel())
y_train
X
corpus = []

for i in range(len(X)):

    cleanr = re.compile('<.*?>')

    review = re.sub(cleanr, '', X[i][0])

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review  if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
corpus
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))

X = tfidf.fit_transform(corpus).toarray()
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y.ravel())
# 1st model Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.01,0.1,1, 10, 100], 'penalty': ['l2'],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag']}]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X, y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_
best_accuracy



best_parameters 
classifier = LogisticRegression(random_state = 0,C= 100, penalty = 'l2', solver ='newton-cg')
classifier.fit(X_train_after, y_train)
y_pred = classifier.predict(X_test_after)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
#with penality l1

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.01,0.1,1, 10, 100], 'penalty': ['l1'],'solver':[  'liblinear']}]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X, y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_
best_accuracy
best_parameters
classifier = LogisticRegression(random_state = 0,C= 100, penalty = 'l2', solver ='newton-cg')

classifier.fit(X_train_after, y_train)

y_pred = classifier.predict(X_test_after)
y_train_pred = classifier.predict(X_train_after)
from sklearn.metrics import accuracy_score

test_err= accuracy_score(y_test, y_pred)
model_err= accuracy_score( y_train,y_train_pred)
model_err
test_err