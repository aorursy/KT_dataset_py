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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import string
def extracting_metafeatures1(textcol, trainDF):
    trainDF['char_count'] = trainDF[textcol].apply(len)
    trainDF['word_count'] = trainDF[textcol].apply(lambda x: len(x.split()))
    #trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
    trainDF['punctuation_count'] = trainDF[textcol].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
    trainDF['title_word_count'] = trainDF[textcol].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = trainDF[textcol].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words='english')
X_train_dtm = tfidf.fit_transform(train.question_text)
X_test_dtm = tfidf.transform(test.question_text)
X_test_dtm
extracting_metafeatures1('question_text',train)
extracting_metafeatures1('question_text',test)
print(train.head())
print(test.head())
sns.boxplot(x = 'target', y = 'char_count', data = train)
sns.boxplot(x = 'target', y = 'word_count', data = train)
sns.boxplot(x = 'target', y = 'punctuation_count', data = train)
sns.boxplot(x = 'target', y = 'title_word_count', data = train)
sns.boxplot(x = 'target', y = 'upper_case_word_count', data = train)
print(X_train_dtm.shape)
print(X_test_dtm.shape)
target = train.target
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled_train = scaler.fit_transform(train[['char_count','word_count','upper_case_word_count','punctuation_count','title_word_count']])
scaled_test = scaler.fit_transform(test[['char_count','word_count','upper_case_word_count','punctuation_count','title_word_count']])
scaled_train
from scipy.sparse import coo_matrix, hstack
A = X_train_dtm
B = coo_matrix(scaled_train)

print(A.shape)
print(B.shape)
newtrain = hstack([A,B])
print(newtrain.shape)

C = X_test_dtm
D = coo_matrix(scaled_test)

print(C.shape)
print(D.shape)
newtest = hstack([C,D])
print(newtest.shape)
#Linear SVC
print("\nLinear SVC")
logreg = LinearSVC()
# train the model using X_train_dtm
%time logreg.fit(newtrain, train.target)
# make class predictions for X_test_dtm
y_pred = logreg.predict(newtest)
submission = test[['qid','char_count']]
submission.head()

submission['prediction'] = y_pred
submission.prediction = submission.prediction.astype(int)
submission.drop('char_count', axis = 1, inplace = True)
submission.head()
submission.to_csv('submission.csv', index=False)
