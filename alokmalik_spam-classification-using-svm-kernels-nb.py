# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from gensim import parsing
from sklearn.metrics import accuracy_score
import chardet
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#detects encoding of csv file
with open('../input/spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
    
#put csv file in a dataframe.
df = pd.read_csv("../input/spam.csv", encoding = result['encoding'])


df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df['v1'] = df.v1.map({'ham':0, 'spam':1})

#df['v1'] = df.v1.map({'ham':0, 'spam':1})
# Any results you write to the current directory are saved as output.
#Count observations in each label
df.v1.value_counts()
def parse(s):
    parsing.stem_text(s)
    return s

#applying parsing to comments.
for i in range(0,len(df)):
    df.iloc[i,1]=parse(df.iloc[i,1])
    df.iloc[i,1]=df.iloc[i,1].lower()
X, y = df['v2'].tolist(), df['v1'].tolist()

#Train and test set split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Multinomial NB is the type of Naive Bayes which is often used to text classification.
#for more info about multinomial Naive bayes check out http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
#Count vectorizer create a matrix of all SMS where each value represents the number of times the corresponding word appeared in that sms.
#For more info on Count Vectorizer check out http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#and https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
#tf–idf or TFIDF, short for term frequency–inverse document frequency
#is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
#TFIDF significantly improves accuracy of a text classifier.
#For more info on why we use TFIDF check out https://en.wikipedia.org/wiki/Tf%E2%80%93idf

#Use pipeline to carry out steps in sequence with a single object, this time we'll use Multinomial NB classifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

#train model
text_clf.fit(X_train, y_train)


#predict class form test data 
predicted = text_clf.predict(X_test)

print(accuracy_score(y_test, predicted))
print(roc_auc_score(y_test,predicted))
print(confusion_matrix(y_test, predicted))
#As you can see above, multinomial NB gives 96.70% accuracy
#Use pipeline to carry out steps in sequence with a single object, this time we'll use Multinomial NB classifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression())])

#train model
text_clf.fit(X_train, y_train)


#predict class form test data 
predicted = text_clf.predict(X_test)

print(accuracy_score(y_test, predicted))
print(roc_auc_score(y_test,predicted))
print(confusion_matrix(y_test, predicted))
#As you can see above, multinomial NB gives 96.70% accuracy
#Use pipeline to carry out steps in sequence with a single object, this time we'll use SVM with gaussian kernel
# for more info on different kernels check out http://scikit-learn.org/stable/modules/svm.html
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

#train model
text_clf.fit(X_train, y_train)

#predict class form test data 
predicted = text_clf.predict(X_test)

print(accuracy_score(y_test, predicted))
print(roc_auc_score(y_test,predicted))
print(confusion_matrix(y_test, predicted))

#Use pipeline to carry out steps in sequence with a single object, this time we'll use SVM classifier with polynomial kernel
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='poly'))])

#train model
text_clf.fit(X_train, y_train)

#predict class form test data 
predicted = text_clf.predict(X_test)

print(accuracy_score(y_test, predicted))
print(roc_auc_score(y_test,predicted))
print(confusion_matrix(y_test, predicted))
#SVM with polynomial kernel gives only 87.72% accuracy, which is same as gaussian kernel. Uptil now Multinomial NB has given highest
#accuracy, i.e. 96.70%. Let's see if linear kernel in SVM is able to cross that score.
#Linear kernel is considered most suitable for text classification
#Use pipeline to carry out steps in sequence with a single object, this time we'll use SVM classifier with linear kernel
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='linear'))])

#train model
text_clf.fit(X_train, y_train)

#predict class form test data 
predicted = text_clf.predict(X_test)

print(accuracy_score(y_test, predicted))
print(roc_auc_score(y_test,predicted))
print(confusion_matrix(y_test, predicted))