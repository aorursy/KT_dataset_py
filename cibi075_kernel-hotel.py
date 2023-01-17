import os

import re

import nltk

import numpy as np

import pandas as pd





train = pd.read_csv("../input/Hotel.csv",encoding = "ISO-8859-1")
x=train['reviews.text']

y=train['reviews.rating']
x.shape
from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(x, y, test_size=0.3)





def preprocess(data):

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean(x, punct))

    return data
x=preprocess(x)
#test

from sklearn.feature_extraction.text import TfidfVectorizer



# create the transform

vectorizer = TfidfVectorizer()

# tokenize and build vocab

vectorizer.fit_transform(x.astype('U'))

#vectorizer.fit(x)

# summarize

print(vectorizer.vocabulary_)

print(vectorizer.idf_)

# encode document

x_train = vectorizer.fit_transform(x.astype('U'))

# summarize encoded vector

print(x_train.shape)

#vector=vector.toarray()

#vector=np.resize(21,207)
x_test=preprocess(x_test)

from sklearn.feature_extraction.text import TfidfVectorizer



# create the transform

##vectorizer = TfidfVectorizer()

# tokenize and build vocab

##vectorizer.transform(x_test.astype('U'))

#vectorizer.fit(x)

# summarize

##print(vectorizer.vocabulary_)

##print(vectorizer.idf_)

# encode document

x_test = vectorizer.transform(x_test.astype('U'))

# summarize encoded vector

print(x_test.shape)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
y_train = np.asarray(y, dtype="int64")
y_train.shape
#test

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

clf.fit(x_train, y_train)  

predictions=clf.predict(x_test)
#test

from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(x_train, y_train) 

predictions=clf.predict(x_test)
#test

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2,

                             random_state=0)

clf.fit(x_train, y_train)  



predictions=clf.predict(x_test)
#test

from sklearn import tree



clf = tree.DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)

predictions=clf.predict(x_test)
#test

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=60,algorithm="auto")

model=neigh.fit(x_train, y_train) 



predictions = model.predict(x_test)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
y_test = np.asarray(y_test, dtype="int64")
from sklearn.metrics import accuracy_score



accuracy_score(y_test, predictions)

submission = pd.read_csv('../input/submission.csv')



submission['prediction'] = predictions





submission.to_csv('submission.csv',index=False)