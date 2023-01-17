import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

import glob

import random



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score 



from nltk.corpus import stopwords

from nltk.tokenize import PunktSentenceTokenizer



# Materials partially adapted from https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments
df = pd.read_csv('../input/dataset.csv', encoding="ISO-8859-1")
df.head()
df['Sentiment'].value_counts()
random.sample(df['SentimentText'].values.tolist(), 1)
corpus = ["He is ::having a great Time, at the park time?",

          "She, unlike most women, is a big player on the park's grass.",

          "she can't be going"]



vectorizer = CountVectorizer(stop_words="english", analyzer='word', 

                             ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)



count_train = vectorizer.fit(corpus)

corpus_vectorized = vectorizer.transform(corpus)



print("Vocabulary content:")

print(vectorizer.get_feature_names())

print("Vectorized corpus:")

print(corpus_vectorized.toarray())
corpus = ["He is ::having a great Time, at the park time?",

          "She, unlike most women, is a big player on the park's grass.",

          "she can't be going"]



vectorizer = CountVectorizer(stop_words="english", analyzer='word', 

                             ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None)



count_train = vectorizer.fit(corpus)

corpus_vectorized = vectorizer.transform(corpus)



print("Vocabulary content:")

print(vectorizer.get_feature_names())

print("Vectorized corpus:")

print(corpus_vectorized.toarray())
corpus = ["He is ::having a great Time, at the park time?",

          "She, unlike most women, is a big player on the park's grass.",

          "she can't be going"]



vectorizer = CountVectorizer(stop_words="english", analyzer='word', 

                             ngram_range=(1, 3), max_df=1.0, min_df=1, max_features=None)



count_train = vectorizer.fit(corpus)

corpus_vectorized = vectorizer.transform(corpus)



print("Vocabulary content:")

print(vectorizer.get_feature_names())

print("Vectorized corpus:")

print(corpus_vectorized.toarray())
corpus = ['His smile was not perfect',

          'His smile was not not not not perfect',

          'she not sang']



vectorizer = TfidfVectorizer()

count_train = vectorizer.fit(corpus)

corpus_vectorized = vectorizer.transform(corpus)



print("Vocabulary content:")

print(vectorizer.get_feature_names())

print("Vectorized corpus:")

print(corpus_vectorized.toarray())
idf = vectorizer.idf_

vocabs = vectorizer.get_feature_names()

rr = dict(zip(vocabs, idf))



token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()

token_weight.columns=('token','weight')

token_weight = token_weight.sort_values(by='weight', ascending=False)

token_weight 



sns.barplot(x='token', y='weight', data=token_weight)            

plt.title("Inverse Document Frequency(idf) per token")

fig=plt.gcf()

fig.set_size_inches(20, 10)

plt.show()
x_train, x_test, y_train, y_test = train_test_split(df['SentimentText'], df['Sentiment'], test_size=0.2)



cvect = CountVectorizer(ngram_range=(1, 2))

cvect.fit(x_train)

x_train_vectorized = cvect.transform(x_train)

x_test_vectorized = cvect.transform(x_test)
sgd = SGDClassifier()

sgd.fit(x_train_vectorized, y_train)

y_pred = sgd.predict(x_test_vectorized)

print("Accuracy score using CountVectorizer and SGD Classifier: {}".format(accuracy_score(y_pred, y_test)))
nb = MultinomialNB()

nb.fit(x_train_vectorized, y_train)

y_pred = nb.predict(x_test_vectorized)

print("Accuracy score using CountVectorizer and Multinomial Naive Bayes: {}".format(accuracy_score(y_pred, y_test)))
for i, (x, y) in enumerate(zip(x_test.values, y_test)):

    print('-----------------------------------------')

    print("Text:")

    print(x)

    print("Sentiment:")

    print(y)



    if i == 30:

        break

    
x_train, x_test, y_train, y_test = train_test_split(df['SentimentText'], df['Sentiment'], test_size=0.2)



vect = TfidfVectorizer(ngram_range=(1, 2))

vect.fit(x_train)

x_train_vectorized = vect.transform(x_train)

x_test_vectorized = vect.transform(x_test)
idf = vect.idf_

vocabs = vect.get_feature_names()

rr = dict(zip(vocabs, idf))



token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()

token_weight.columns=('token','weight')

token_weight = token_weight.sort_values(by='weight', ascending=False)
sns.barplot(x='token', y='weight', data=token_weight.iloc[:10])            

plt.title("Inverse Document Frequency(idf) per token")

fig=plt.gcf()

fig.set_size_inches(20, 10)

plt.show()
token_weight.head(30)
sns.barplot(x='token', y='weight', data=token_weight.iloc[-10:])            

plt.title("Inverse Document Frequency(idf) per token")

fig=plt.gcf()

fig.set_size_inches(20, 10)

plt.show()
token_weight.tail(30)
sgd = SGDClassifier()

sgd.fit(x_train_vectorized, y_train)

y_pred = sgd.predict(x_test_vectorized)

print("Accuracy score using TFIDF and SGD Classifier: {}".format(accuracy_score(y_pred, y_test)))
nb = MultinomialNB()

nb.fit(x_train_vectorized, y_train)

y_pred = nb.predict(x_test_vectorized)

print("Accuracy score using TFIDF and Multinomial Naive Bayes: {}".format(accuracy_score(y_pred, y_test)))
for i, (x, y) in enumerate(zip(x_test.values, y_test)):

    print('-----------------------------------------')

    print("Text:")

    print(x)

    print("Sentiment:")

    print(y)



    if i == 30:

        break