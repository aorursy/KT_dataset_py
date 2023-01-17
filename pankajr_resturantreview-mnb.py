# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Load data set

df = pd.read_csv("/kaggle/input/mydata/restaurantreviews.csv", sep="\t", names=['Review','Liked'], encoding="latin-1")
# Print all data

df = df[1:]

df
# Rows and columns

df.shape
# Rows

df.shape[0]
# columns

df.shape[1]
#First 5 rows

df.head(5)
#Last 5 records

df.tail(5)
# filter data

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()

wordnet=WordNetLemmatizer()

corpus = []

for i in range(0, len(df)):

    review = re.sub('[^a-zA-Z]', ' ', df['Review'].values[i])

    review = review.lower()

    review = review.split()

    

    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
# Vectorize setences and define x and y

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2500)

x = cv.fit_transform(corpus).toarray()

y = df['Liked'].values
#Split test train data into 70:30

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
#Prepare Multinomial Naive Bayes model

from sklearn.naive_bayes import MultinomialNB
#fit the data

detect_model = MultinomialNB().fit(x_train, y_train)
#Predict result

y_pred=detect_model.predict(x_test)
#classification Report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
#Confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
#Accuracy score

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)