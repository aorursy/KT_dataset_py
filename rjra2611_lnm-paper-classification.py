import re

import pandas as pd # CSV file I/O (pd.read_csv)

from nltk.corpus import stopwords

import numpy as np

import sklearn

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score ,confusion_matrix
df = pd.read_excel("../input/LNM_paper_ocred_data.xlsx") #Importing data from CSV

# news = (news.loc[news['CATEGORY'].isin(['b','e'])]) #Retaining rows that belong to categories 'b' and 'e'

df.head()
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(df["CLEAN_DATA"], df["FIELD_BRANCH"], test_size = 0.2)

X_train = np.array(X_train);

X_test = np.array(X_test);

Y_train = np.array(Y_train);

Y_test = np.array(Y_test);

cleanHeadlines_train = [] #To append processed headlines

cleanHeadlines_test = [] #To append processed headlines

number_reviews_train = len(X_train) #Calculating the number of reviews

number_reviews_test = len(X_test) #Calculating the number of reviews
cleanHeadlines_train=X_train

cleanHeadlines_test=X_test
vectorize = sklearn.feature_extraction.text.CountVectorizer(analyzer = "word",max_features = 1700)

bagOfWords_train = vectorize.fit_transform(cleanHeadlines_train)

X_train = bagOfWords_train.toarray()
bagOfWords_test = vectorize.transform(cleanHeadlines_test)

X_test = bagOfWords_test.toarray()
vocab = vectorize.get_feature_names()

nb = MultinomialNB()

nb.fit(X_train, Y_train)

print(nb.score(X_test, Y_test))
logistic_Regression = LogisticRegression()

logistic_Regression.fit(X_train,Y_train)

Y_predict = logistic_Regression.predict(X_test)

print(accuracy_score(Y_test,Y_predict))
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC
model = LinearSVC()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print(accuracy_score(Y_test,Y_predict))
model = RandomForestClassifier()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print(accuracy_score(Y_test,Y_predict))