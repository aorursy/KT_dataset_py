# Importing the libraries

import numpy as np

import pandas as pd

import itertools

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
#Importing the datasets

df = pd.read_csv('/kaggle/input/news.csv')

#Get shape and head

print(df.shape)

df.head()
# Cheching for missing vaslues

df.isnull().sum()
df = df.drop(columns = ['Unnamed: 0'])

df
df['label'] = df['label'].astype('category')

df['label'] = df['label'].cat.codes

df
df['title_length'] = df['title'].apply(len)

df
df['text_length'] = df['text'].apply(len)

df
#DataFlair - Get the labels

label=df.label

label.head()
df.describe().astype(int)
df.hist(column='title_length', by='label')

# x axis is the title length

# y axis is the number of news
df.hist(column='text_length', by='label')

# x axis is the text length

# y axis is the number of news
# DataFlair - Splitting the dataset

x_train, x_test, y_train, y_test = train_test_split(df['text'], label, test_size=0.2, random_state=7)
#DataFlair - Initialize a TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)



#DataFlair - Fit and transform train set, transform test set

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 

tfidf_test=tfidf_vectorizer.transform(x_test)
#DataFlair - Initialize a PassiveAggressiveClassifier

pac=PassiveAggressiveClassifier(max_iter=50)

pac.fit(tfidf_train,y_train)



#DataFlair - Predict on the test set and calculate accuracy

y_pred=pac.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

print(f'Accuracy: {round(score*100,2)}%')
#DataFlair - Build confusion matrix

confusion_matrix(y_test,y_pred, labels=[0,1])