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
data = pd.read_csv("../input/all_tickets.csv")

data.info()
numpy_array = data.as_matrix()

X = numpy_array[:,1]

Y = numpy_array[:,2]

Y=Y.astype('int')
print('***Without Stemming***')
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB



X_train, X_test, y_train, y_test = train_test_split(

    X, Y, test_size=0.2, random_state=42)





from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

X_train_counts.shape
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, y_train)
X_test_counts = count_vect.transform(X_test)

X_test_counts.shape

x_test_tfidf = tfidf_transformer.transform(X_test_counts)

x_test_tfidf.shape

predicted = clf.predict(x_test_tfidf)

round(np.mean(predicted == y_test), 4)
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier().fit(X_train_tfidf, y_train)

predicted1 = sgd_clf.predict(x_test_tfidf)

round(np.mean(predicted1 == y_test), 4)
print('***Stemming***')



import nltk



from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

X_train_counts = stemmed_count_vect.fit_transform(X_train)



X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)



clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_test_counts = stemmed_count_vect.transform(X_test)

print(X_test_counts.shape)

x_test_tfidf = tfidf_transformer.transform(X_test_counts)

print(x_test_tfidf.shape)



predicted = clf.predict(x_test_tfidf)

print('Multinomial NB score ', round(np.mean(predicted == y_test), 4))





sgd_clf = SGDClassifier().fit(X_train_tfidf, y_train)

predicted1 = sgd_clf.predict(x_test_tfidf)

print('SGDClassifier score ' , round(np.mean(predicted1 == y_test), 4))