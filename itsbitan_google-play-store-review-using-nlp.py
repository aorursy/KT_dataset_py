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

#Importing the Dataset

df =pd.read_csv("../input/googleplaystore_user_reviews.csv",encoding="latin1")
#Now Lets set dataset which collumns we are interested

df = pd.concat([df.Translated_Review, df.Sentiment], axis = 1)
#Now eleminate the nan value becasue they can affect our model

df.dropna(axis = 0, inplace = True)
#Replace the Sentiment by Encoding, Positive=0, Negative = 1, Netural= 2

df.Sentiment = [0 if i=="Positive" else 1 if i== "Negative" else 2 for i in df.Sentiment]
#Now lets Cleaning the Text

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

text_list = []

for i in df.Translated_Review :

    review = re.sub('[^a-zA-Z]', ' ', i)

    review = review.lower() 

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review) 

    text_list.append(review)  
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1000)

x = cv.fit_transform(text_list).toarray()

y = df.iloc[:, 1].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
# Now Fitting Naive Bayes classifier to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(x_test)

# Making the Confusion Matrix and find Accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)    

accuracy = accuracy_score(y_test, y_pred)

print(cm)

print(accuracy)
#Now Fitting Random Forest Classifier to the Traning set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)

classifier.fit(x_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(x_test)
# Making the Confusion Matrix and find Accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)    

accuracy = accuracy_score(y_test, y_pred)

print(cm)

print(accuracy)