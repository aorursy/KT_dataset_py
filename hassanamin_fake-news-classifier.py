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
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import numpy as np

import itertools

from sklearn.linear_model import PassiveAggressiveClassifier



df = pd.read_csv('../input/fake_or_real_news.csv') # Load data into DataFrame

y = df.label

X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33,random_state=53)

from sklearn.feature_extraction.text import CountVectorizer

# list of text documents

text = ["The quick brown fox jumped over the lazy dog."]

# create the transform

vectorizer = CountVectorizer()

# tokenize and build vocab

vectorizer.fit(text)

# summarize

print(vectorizer.vocabulary_)

# encode document

vector = vectorizer.transform(text)

# summarize encoded vector

print(vector.shape)

print(type(vector))

print(vector.toarray())
# encode another document

text2 = ["the puppy"]

vector = vectorizer.transform(text2)

print(vector.toarray())
corpus = [

'This is the first document.',

'This document is the second document.',

'And this is the third one.',

'Is this the first document?',

]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())

print(X.toarray())  
count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train.values)

count_test = count_vectorizer.transform(X_test.values)
from sklearn.feature_extraction.text import TfidfVectorizer



# list of text documents

text = ["The quick brown fox jumped over the lazy dog.","The dog.", "The fox"]



# create the transform

vectorizer = TfidfVectorizer()



# tokenize and build vocab

vectorizer.fit(text)



# summarize

print("Vocabulary :- ",vectorizer.vocabulary_)

print("IDF :- ",vectorizer.idf_)



# encode document

vector = vectorizer.transform([text[0]])



# summarize encoded vector

print("Text : ", text[0])

print("Shape : ",vector.shape)



print("Representation : ", vector.toarray())
# Initialize the `tfidf_vectorizer` 

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 



# Fit and transform the training data 

tfidf_train = tfidf_vectorizer.fit_transform(X_train) 



# Transform the test set 

tfidf_test = tfidf_vectorizer.transform(X_test)



print(tfidf_test)
# Get the feature names of `tfidf_vectorizer` 

print(tfidf_vectorizer.get_feature_names()[-10:])

# Get the feature names of `count_vectorizer` 

print(count_vectorizer.get_feature_names()[0:10])
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    See full source and example: 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
clf = MultinomialNB() 

clf.fit(count_train, y_train)

pred = clf.predict(count_test)

score = accuracy_score(y_test, pred)

print("accuracy:   %0.3f" % score)

cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])

print(cm)

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
clf = MultinomialNB() 

clf.fit(tfidf_train, y_train)

pred = clf.predict(tfidf_test)

score = accuracy_score(y_test, pred)

print("accuracy:   %0.3f" % score)

cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])

print(cm)

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
from sklearn.feature_extraction.text import HashingVectorizer

# list of text documents

text = ["The quick brown fox jumped over the lazy dog."]

# create the transform

vectorizer = HashingVectorizer(n_features=20)

# encode document

vector = vectorizer.transform(text)

# summarize encoded vector

print(vector.shape)

print(vector.toarray())


from sklearn.feature_extraction.text import HashingVectorizer

# Initialize the hashing vectorizer

hashing_vectorizer = HashingVectorizer(stop_words='english',n_features=5000,non_negative=True)



# Fit and transform the training data 

hashing_train = hashing_vectorizer.fit_transform(X_train)



# Transform the test set 

hashing_test = hashing_vectorizer.transform(X_test)



print(hashing_test)



clf = MultinomialNB() 

clf.fit(hashing_train, y_train)

pred = clf.predict(hashing_test)

score = accuracy_score(y_test, pred)

print("accuracy:   %0.3f" % score)

cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])

print(cm)

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])