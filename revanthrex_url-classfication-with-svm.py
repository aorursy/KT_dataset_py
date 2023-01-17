import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import train_test_split

%matplotlib inline 

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import chi2

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.utils import np_utils

import re

from keras.preprocessing import sequence

from keras.preprocessing.text import one_hot

from keras.preprocessing.text import text_to_word_sequence

from sklearn.svm import LinearSVC
names=['URL','Category']

df=pd.read_csv('../input/URL Classification.csv',names=names, na_filter=False)

df1 = df[1:2001]

df2 = df[50000:52000]

df3 = df[520000:522000]

df4 =df[535300:537300]

df5 = df[650000:652000]

df6= df[710000:712000]

df7=  df[764200:766200]

df8=  df[793080:795080]

df9=  df[839730:841730]

df10=  df[850000:852000]

df11=  df[955250:957250]

df12=  df[1013000:1015000]

df13=  df[1143000:1145000]

df14=  df[1293000:1295000]

df15=  df[1492000:1494000]

dt=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], axis=0)

df.drop(df.index[1:2000],inplace= True)

df.drop(df.index[50000:52000],inplace= True)

df.drop(df.index[520000:522000],inplace= True)

df.drop(df.index[535300:537300],inplace= True)

df.drop(df.index[650000:652000],inplace= True)

df.drop(df.index[710000:712000],inplace= True)

df.drop(df.index[764200:766200],inplace= True)

df.drop(df.index[793080:795080],inplace= True)

df.drop(df.index[839730:841730],inplace= True)

df.drop(df.index[850000:852000],inplace= True)

df.drop(df.index[955250:957250],inplace= True)

df.drop(df.index[1013000:1015000],inplace= True)

df.drop(df.index[1143000:1145000],inplace= True)

df.drop(df.index[1293000:1295000],inplace= True)

df.drop(df.index[1492000:1494000],inplace= True)

df.tail()
dt.Category.value_counts().plot(figsize=(12,5),kind='bar',color='green');

plt.xlabel('Category')

plt.ylabel('Total Number Of Individual Category for Testing')
df.Category.value_counts().plot(figsize=(12,5),kind='bar',color='green');

plt.xlabel('Category')

plt.ylabel('Total Number Of Individual Category for Training')
X_train=df['URL']

y_train=df['Category']

print(X_train)

X_train.shape
X_test=dt['URL']

y_test=dt['Category']

print(X_test)

X_test.shape
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LinearSVC())])

text_clf = text_clf.fit(X_train, y_train)
from sklearn.model_selection import RandomizedSearchCV

n_iter_search = 5

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf__C': [0.1, 1,10,100]}

gs_clf = RandomizedSearchCV(text_clf, parameters, n_iter = n_iter_search)

gs_clf = gs_clf.fit(X_train, y_train)
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix

y_pred=gs_clf.predict(X_test)

print(y_pred)

precision_recall_fscore_support(y_test, y_pred, average='weighted')
y_pred=gs_clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred,digits=4))
print('Naive Bayes Test Accuracy = ',metrics.accuracy_score(y_test,y_pred))