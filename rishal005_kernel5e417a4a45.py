# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

spam = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding="latin-1")

spam.head()
spam.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
spam.columns=['Labels','Message']
spam.head()
spam.shape
# examine the class distribution

spam.Labels.value_counts()
# convert label to a numerical variable

spam['label_num'] = spam.Labels.map({'ham':0, 'spam':1})
spam.head()
X = spam.Message

y = spam.label_num
# split X and y into training and testing sets

# by default, it splits 75% training and 25% test

# random_state=1 for reproducibility

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# 1. import and instantiate CountVectorizer (with the default parameters)

from sklearn.feature_extraction.text import CountVectorizer



# 2. instantiate CountVectorizer (vectorizer)

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm
# 4. transform testing data (using fitted vocabulary) into a document-term matrix

X_test_dtm = vect.transform(X_test)

X_test_dtm



# you can see that the number of columns, 7456, is the same as what we have learned above in X_train_dtm
# 1. import

from sklearn.naive_bayes import MultinomialNB



# 2. instantiate a Multinomial Naive Bayes model

nb = MultinomialNB()
# 3. train the model 



nb.fit(X_train_dtm, y_train)
# 4. make class predictions for X_test_dtm

y_pred_class = nb.predict(X_test_dtm)
# calculate accuracy of class predictions

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
# print the confusion matrix

metrics.confusion_matrix(y_test, y_pred_class)
# print message text for the false positives (ham incorrectly classified as spam)



X_test[(y_pred_class==1) & (y_test==0)]
# print message text for the false negatives (spam incorrectly classified as ham)

X_test[(y_pred_class==0) & (y_test==1)]
# example false negative

X_test[3132]
# calculate AUC

y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]

metrics.roc_auc_score(y_test, y_pred_prob)
# 1. import

from sklearn.linear_model import LogisticRegression



# 2. instantiate a logistic regression model

logreg = LogisticRegression()
# 3. train the model using X_train_dtm

logreg.fit(X_train_dtm, y_train)
# 4. make class predictions for X_test_dtm

y_pred_class = logreg.predict(X_test_dtm)
# calculate predicted probabilities for X_test_dtm (well calibrated)

y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

y_pred_prob
# calculate accuracy

metrics.accuracy_score(y_test, y_pred_class)
# calculate AUC

metrics.roc_auc_score(y_test, y_pred_prob)
# Without removing  English stop words

vect1 = CountVectorizer()



X_train_1 = vect1.fit_transform(X_train)



X_train_1
# remove English stop words

vect1 = CountVectorizer(stop_words='english')



X_train_1 = vect1.fit_transform(X_train)



X_train_1
# include 1-grams and 2-grams



# how to differentiate between "Happy", "Not Happy", "Very Happy"

vect2 = CountVectorizer(ngram_range=(1, 2))



X_train_2 = vect2.fit_transform(X_train)



X_train_2
# ignore terms that appear in more than 50% of the documents

vect3 = CountVectorizer(max_df=0.5)



X_train_3 = vect3.fit_transform(X_train)



X_train_3
# only keep terms that appear in at least 2 documents

vect4 = CountVectorizer(min_df=2)



X_train_4 = vect4.fit_transform(X_train)



X_train_4
vect_combined= CountVectorizer(stop_words='english',ngram_range=(1, 2),min_df=2,max_df=0.5)
X_train_c = vect_combined.fit_transform(X_train)

X_test_c = vect_combined.transform(X_test)



X_train_c
# 1. import

from sklearn.naive_bayes import MultinomialNB



# 2. instantiate a Multinomial Naive Bayes model

nb = MultinomialNB()



nb.fit(X_train_c, y_train)



y_pred_class = nb.predict(X_test_c)



metrics.confusion_matrix(y_test, y_pred_class)