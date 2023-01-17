# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import datetime 

import calendar

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", context="talk")

import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer 

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn import decomposition

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df_fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

df_real = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
df_fake["RESULT"] = 0

df_real["RESULT"] = 1
df = pd.concat([df_fake,df_real],ignore_index=True)
df
df["Final_Text"] = df["title"] + df["text"]
df["RESULT"].value_counts()
df[['RESULT','subject','Final_Text']].groupby(['RESULT','subject']).count()
lemmatizer = WordNetLemmatizer()

porterstem = PorterStemmer()

lst_lemmetized = []

lst_stemmed = []

def Preprocessing_Lemmatizing_Stemming(text):

    tokens = word_tokenize(text)

    tokens = [w.lower() for w in tokens]

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]

    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]

    words_lemmtized = [lemmatizer.lemmatize(w) for w in words]

    words_stemmed = [porterstem.stem(w) for w in words]

    

    lst_lemmetized.append(" ".join(words_lemmtized))

    lst_stemmed.append(" ".join(words_stemmed))
for i in df["Final_Text"]:

    Preprocessing_Lemmatizing_Stemming(str(i))
X = lst_lemmetized

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', LogisticRegression())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Lemmetized text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))





X = lst_stemmed

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)





pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', LogisticRegression())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Stemmed text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))
X = lst_lemmetized

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', MultinomialNB())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Lemmetized text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))





X = lst_stemmed

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)





pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', MultinomialNB())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Stemmed text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))
X = lst_lemmetized

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', KNeighborsClassifier())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Lemmetized text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))





X = lst_stemmed

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)





pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', KNeighborsClassifier())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Stemmed text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))
X = lst_lemmetized

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', RandomForestClassifier())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Lemmetized text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))





X = lst_stemmed

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)





pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', RandomForestClassifier())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Stemmed text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))
X = lst_lemmetized

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', DecisionTreeClassifier())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Lemmetized text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))





X = lst_stemmed

y = df["RESULT"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)





pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', DecisionTreeClassifier())])



model = pipe.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy of the model using Stemmed text {}%".format(round(accuracy_score(y_test, y_pred)*100,2)))

print(confusion_matrix(y_test,y_pred))
X = lst_lemmetized

y = df["RESULT"]







pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', DecisionTreeClassifier())])



criterion = ['gini', 'entropy']

max_depth = [2,4,6,8]



parameters = dict(model__criterion=criterion, model__max_depth=max_depth)



clf = GridSearchCV(pipe, parameters)



clf.fit(X, y)
print("Grid Search results")

print('Criterion:', clf.best_estimator_.get_params()['model__criterion'])

print('Max_depth:', clf.best_estimator_.get_params()['model__max_depth'])