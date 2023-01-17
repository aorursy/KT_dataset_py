# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Devotion_Reviews.csv')

df_review = df[['text', 'recommended']].copy()

df_review['recommended'] = df['recommended'].astype(dtype=np.int64)

df_review.head()
# remove Chinese character

printable = set(string.printable)

df_review['cleantext'] = df['text'].apply(lambda row: ''.join(filter(lambda x:x in printable,row)))



REPLACE = re.compile('[.;:!\'?,\"()\[\]]')

def pre_process(text):

    # lowercase

    text = text.lower()

    # tags

    text = re.sub('&lt;/?.*?&gt;',' &lt;&gt; ',text)

    # special characters and digits

    text=re.sub('(\\d|\\W)+',' ',text)

    # remove punctuation

    #text = re.sub('[.;:!\'?,\"()\[\]]', '', text)

    #text = [REPLACE.sub('', line) for line in text]

    

    return text



df_review['cleantext'] = df_review['cleantext'].apply(lambda x:pre_process(x))





from nltk.corpus import stopwords

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS



#english_stop_words = stopwords.words('english')

english_stop_words = ENGLISH_STOP_WORDS

def remove_stop_words(corpus):

    removed_stop_words = []

    for review in corpus:

        removed_stop_words.append(

            ' '.join([word for word in review.split() 

                      if word not in english_stop_words])

        )

    return removed_stop_words



df_review['cleantext'] = remove_stop_words(df_review['cleantext'])
print(df_review['text'][10])

print(df_review['cleantext'][10])
# Stemming

from nltk.stem.porter import PorterStemmer



def get_stemmed_text(corpus):

    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]



df_review['stemmedtext'] = get_stemmed_text(df_review['cleantext'])
df_review.head()
# Lemmatization

from nltk.stem import WordNetLemmatizer

def get_lemmatized_text(corpus):

    lemmatizer = WordNetLemmatizer()

    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]



df_review['lemmatext'] = get_lemmatized_text(df_review['stemmedtext'])
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))

ngram_vectorizer.fit(df_review['lemmatext'])

X = ngram_vectorizer.transform(df_review['lemmatext'])

y = df_review['recommended']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    

    lr = LogisticRegression(C=c)

    lr.fit(X_train, y_train)

    print('Accuracy for C=%s: %s' % (c, accuracy_score(y_test, lr.predict(X_test))))

    

final_ngram = LogisticRegression(C=1)

final_ngram.fit(X, y)

print('Final Accuracy: %s' % accuracy_score(y_test, final_ngram.predict(X_test)))
feature_to_coef = {

    word: coef for word, coef in zip(

     ngram_vectorizer.get_feature_names(), final_ngram.coef_[0])

}



print('Positive Words')

for best_positive in sorted(

    feature_to_coef.items(),

    key=lambda x: x[1],

    reverse=True)[:10]:

    print(best_positive)

    

print('Negative Words')

for best_negative in sorted(

    feature_to_coef.items(),

    key=lambda x: x[1])[:10]:

    print(best_negative)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf_vectorizer.fit(df_review['lemmatext'])

X = tfidf_vectorizer.transform(df_review['lemmatext'])

y = df_review['recommended']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    

    lr = LogisticRegression(C=c)

    lr.fit(X_train, y_train)

    print('Accuracy for C=%s: %s' %(c, accuracy_score(y_test, lr.predict(X_test))))

    

final_tfidf = LogisticRegression(C=1)

final_tfidf.fit(X, y)

accuracy_score(y, final_tfidf.predict(X))
feature_to_coef = {

    word: coef for word, coef in zip(

     tfidf_vectorizer.get_feature_names(), final_tfidf.coef_[0])

}



print('Positive Words')

for best_positive in sorted(

    feature_to_coef.items(),

    key=lambda x: x[1],

    reverse=True)[:10]:

    print(best_positive)

    

print('Negative Words')

for best_negative in sorted(

    feature_to_coef.items(),

    key=lambda x: x[1])[:10]:

    print(best_negative)
'dont like' in feature_to_coef
feature_to_coef['dont like']
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2))

tfidf_vectorizer.fit(df_review['lemmatext'])

X = tfidf_vectorizer.transform(df_review['lemmatext'])

y = df_review['recommended']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    

    lr = LogisticRegression(C=c)

    lr.fit(X_train, y_train)

    print('Accuracy for C=%s: %s' %(c, accuracy_score(y_test, lr.predict(X_test))))

    

final_tfidf = LogisticRegression(C=1)

final_tfidf.fit(X, y)

accuracy_score(y, final_tfidf.predict(X))
feature_to_coef = {

    word: coef for word, coef in zip(

     tfidf_vectorizer.get_feature_names(), final_tfidf.coef_[0])

}



print('Positive Words')

for best_positive in sorted(

    feature_to_coef.items(),

    key=lambda x: x[1],

    reverse=True)[:10]:

    print(best_positive)

    

print('Negative Words')

for best_negative in sorted(

    feature_to_coef.items(),

    key=lambda x: x[1])[:10]:

    print(best_negative)