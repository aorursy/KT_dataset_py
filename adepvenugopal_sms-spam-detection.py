import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv('../input/spamraw.csv')

df.head()
#Find count and unique messages count of all the messages

df.describe()
#Extract SPAM messages

spam_messages = df[df["type"]=="spam"]

spam_messages.head() 
#Find count and unique messages count of SPAM messages.

spam_messages.describe()
#Plot the counts of HAM (non SPAM) vs SPAM

df.type.value_counts().plot.bar()
data_train, data_test, labels_train, labels_test = train_test_split(df.text,df.type,test_size=0.2,random_state=0) 
vectorizer = CountVectorizer()

data_train_count = vectorizer.fit_transform(data_train)

data_test_count  = vectorizer.transform(data_test)
clf = MultinomialNB()

clf.fit(data_train_count, labels_train)

predictions = clf.predict(data_test_count)
print ("accuracy_score : ", accuracy_score(labels_test, predictions))
print ("confusion_matrix : \n", confusion_matrix(labels_test, predictions))