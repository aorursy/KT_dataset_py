# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing libraries

import pickle

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC 

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix 
# Reading the data

data = pd.read_csv('../input/RomanUrduData.csv', names = ['Text', 'Sentiment', 'Nan'])

# Dropping the unnecessary column

data.drop(['Nan'], axis = 1, inplace = True)
# Defining dependent and independent variables

X = data['Text']

y = data['Sentiment']
# Encoding labels, using LabelEncoder

senti = ['Neutral', 'Positive', 'Negative']

le = LabelEncoder()

Y = le.fit(senti)

Y = le.transform(y)
# Using Tfidf vectorizer and ngram_range = (1, 1)

vect = TfidfVectorizer(ngram_range = (1, 1))

tfidf_x = vect.fit_transform(data['Text'].values.astype('U'))
# Splitting the data

x_train, x_test, y_train, y_test = train_test_split(tfidf_x, Y, test_size = 0.2, random_state = 1)
# Applying MultinomialNB and using Tfidf vectorizer

mnb_tfidf = MultinomialNB()

mnb_tfidf.fit(x_train, y_train)



# score of the model

mnb_tfidf.score(x_test, y_test)
# Grid search CV for SVC

param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}

gd = GridSearchCV(SVC(), param_grid, cv = 3, refit = True, n_jobs = -1)

gd = gd.fit(x_train, y_train)
print(gd.best_score_)

print(gd.best_estimator_)
# Fitting the best estimator and params 

svm_tfidf = SVC(kernel = 'rbf', C = 1000, gamma = 0.001).fit(x_train, y_train) 

svm_predictions = svm_tfidf.predict(x_test)
# model accuracy for X_test   

svm_tfidf.score(x_test, y_test)