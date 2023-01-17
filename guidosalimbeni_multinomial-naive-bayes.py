import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np 

import pandas as pd
#READING INPUT

data = pd.read_csv("/kaggle/input/spooky-author-identification/train.csv")

data.head()
data['author_num'] = data["author"].map({'EAP':0, 'HPL':1, 'MWS':2})

data.head()
X = data['text']

y = data['author_num']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.feature_extraction.text import CountVectorizer
text=["My name is Paul my life is Jane! And we live our life together" , "My name is Guido my life is Victoria! And we live our life together"]

toy = CountVectorizer(stop_words = 'english')

toy.fit_transform(text)

matrix = toy.transform(text)

features = toy.get_feature_names()

df_res = pd.DataFrame(matrix.toarray(), columns=features)

df_res
vect = CountVectorizer(stop_words = 'english')
X_train_matrix = vect.fit_transform(X_train) 
from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(X_train_matrix, y_train)

print(clf.score(X_train_matrix, y_train))

X_test_matrix = vect.transform(X_test) 

print (clf.score(X_test_matrix, y_test))
predicted_result=clf.predict(X_test_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test,predicted_result))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')



X_train_tfidf = vectorizer.fit_transform(X_train) 

X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB

clf2=MultinomialNB()

clf2.fit(X_train_tfidf, y_train)

print(clf2.score(X_train_tfidf, y_train))

X_test_tfidf = vectorizer.transform(X_test) 

print (clf2.score(X_test_tfidf, y_test))
predicted_result_2=clf2.predict(X_test_tfidf)

from sklearn.metrics import classification_report

print(classification_report(y_test,predicted_result_2))
sample = pd.read_csv("/kaggle/input/spooky-author-identification/sample_submission.csv")

sample.head()
test = pd.read_csv("/kaggle/input/spooky-author-identification/test.csv")

test_matrix = vect.transform(test["text"])

predicted_result = clf.predict_proba(test_matrix)
result=pd.DataFrame()

result["id"]=test["id"]

result["EAP"]=predicted_result[:,0]

result["HPL"]=predicted_result[:,1]

result["MWS"]=predicted_result[:,2]

result.head()
result.to_csv("submission_v1.csv", index=False)