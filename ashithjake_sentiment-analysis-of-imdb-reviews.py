# Import Libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import string

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report,confusion_matrix
movie_review = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

movie_review.head()
plt.figure(figsize=(16,6))

sns.countplot('sentiment',data=movie_review)

plt.show()
movie_review['length'] = movie_review['review'].apply(len)

movie_review['length'].describe()
movie_review.hist(column='length',by='sentiment',bins=30,figsize=(12,6))

plt.show()
#Split the data into training set and test set by 70:30 ratio

review_train,review_test,sentiment_train,sentiment_test = train_test_split(movie_review['review'],movie_review['sentiment'],test_size=0.3)
#Method to remove punctuations, stopwords and each message will be represented as a list of tokens

def text_process(movie_text_data):

    

    #remove punctuations

    nopunc = [char for char in movie_text_data if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    

    #return the word if it does not belong to stopwords of English (common words that does not distinguish features)

    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]
vectorizer_model = CountVectorizer(analyzer=text_process).fit(review_train)
sparse_matrix_train = vectorizer_model.transform(review_train)

sparse_matrix_test = vectorizer_model.transform(review_test)

sparse_matrix_train.shape,sparse_matrix_test.shape
tfidfTransformer_model = TfidfTransformer().fit(sparse_matrix_train)
review_tfidf_train = tfidfTransformer_model.transform(sparse_matrix_train)

review_tfidf_test = tfidfTransformer_model.transform(sparse_matrix_test)
sentiment_analysis_model = MultinomialNB().fit(review_tfidf_train,sentiment_train)
predictions = sentiment_analysis_model.predict(review_tfidf_test)
print(classification_report(sentiment_test,predictions))

print(confusion_matrix(sentiment_test,predictions))