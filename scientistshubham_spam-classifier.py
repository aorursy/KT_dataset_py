#Importing the necessary libraries

import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Data is tab separated
#Putting text and type of message into respective label
url="https://raw.githubusercontent.com/tanmaydn/prml-group007/master/SMSSpamCollection?token=AQ6W2IN5VMP2AULUKFQDI4S7RFKDY"
df=pd.read_csv(url, sep='\t', header=None, names=['Label', 'Text'])
df.value_counts('Label')
df=df.sort_values(by='Label').reset_index(drop=True)
df=df[2500:].reset_index(drop=True)
df
print(len(df))
df.isna().any()
sns.countplot(x='Label', data=df)
pd.value_counts(df['Label'])
#Text Pre processing
Y=df['Label']
X=df['Text']
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=56)
cv1=CountVectorizer()
X_traincv=cv1.fit_transform(X_train, Y_train)

a=X_traincv.toarray()
cv1.get_feature_names()
cv1.inverse_transform(a[1])
cv=CountVectorizer(stop_words='english')
X_traincv=cv.fit_transform(X_train)

X_testcv=cv.transform(X_test)
nb_classifier=MultinomialNB()
nb_classifier.fit(X_traincv, Y_train)
pred=nb_classifier.predict(X_testcv)
len(pred)
pd.DataFrame(pred).value_counts()
Y_test.value_counts()
print(metrics.accuracy_score(Y_test, pred))
metrics.confusion_matrix(Y_test, pred, labels=['ham', 'spam'])
from sklearn.feature_extraction.text import TfidfVectorizer
#Initializing the Vectorizer
tfidf=TfidfVectorizer()

X_train_tf=tfidf.fit_transform(X_train)
X_test_tf=tfidf.transform(X_test)
nb_classifier_new=MultinomialNB()

nb_classifier_new.fit(X_train_tf, Y_train)
pred=nb_classifier_new.predict(X_test_tf)
pd.DataFrame(pred).value_counts()
metrics.accuracy_score(Y_test, pred)
metrics.confusion_matrix(Y_test, pred, labels=['ham', 'spam'])