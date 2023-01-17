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
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
!ls ../
train = pd.read_csv('../input/sentiment-analysis-classification/review_train.csv')

test = pd.read_csv('../input/sentiment-analysis-classification/review_test.csv')
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")

REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")



def preprocess_reviews(seq):

    seq = REPLACE_NO_SPACE.sub("", seq.lower())

    seq = REPLACE_WITH_SPACE.sub(" ", seq)

    

    return seq
train['Text'] = train['Text'].map(preprocess_reviews)
train['Text']
test['Text'] = test['Text'].map(preprocess_reviews)
cv = CountVectorizer(binary=True)

cv.fit(train['Text'])



X_train = cv.transform(train['Text'])

X_test = cv.transform(test['Text'])
X_train.shape
y_train = train['Sentiment'].values

y_test = test['Sentiment'].values
X_train_sv, X_val, y_train_sv, y_val = train_test_split(X_train, y_train, test_size=0.33)
print(X_train_sv.shape, X_val.shape, y_train_sv.shape, y_val.shape)
for c in [0.01, 0.05, 0.25, 0.5, 1]:

    lr = LogisticRegression(C=c)

    lr.fit(X_train_sv, y_train_sv)

    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, lr.predict(X_val))))
final_model = LogisticRegression(C=0.5)

final_model.fit(X_train, y_train)

print('Final Accuracy: {}'.format(accuracy_score(y_test, final_model.predict(X_test))))
feature_to_coef = {

    word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])

}



for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:

    print(best_positive)



print('*********************************************************************')



for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=False)[:5]:

    print(best_negative)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):

    removed_stop_words = []

    for review in corpus:

        removed_stop_words.append(' '.join([word for word in review.split() if word not in english_stop_words]))

    return removed_stop_words



train['Text'] = remove_stop_words(train['Text'])
train['Text'][0]
test['Text'] = remove_stop_words(test['Text'])
lemmatizer = WordNetLemmatizer()

def get_lemmatized_text(corpus):

    lemmatized = []

    for review in corpus:

        lemmatized.append(' '.join([lemmatizer.lemmatize(word) for word in review.split()]))

    return lemmatized



train['Text'] = get_lemmatized_text(train['Text'])

test['Text'] = get_lemmatized_text(test['Text'])



train['Text'][0]
unigram_vectorizer = CountVectorizer(binary=True) 

unigram_vectorizer.fit(train['Text'])



X_train = unigram_vectorizer.transform(train['Text'])

X_test = unigram_vectorizer.transform(test['Text'])

print(X_train.shape)



y_train = train['Sentiment'].values

y_test = test['Sentiment'].values



X_train_sv, X_val, y_train_sv, y_val = train_test_split(X_train, y_train, test_size=0.33)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    lr = LogisticRegression(C=c)

    lr.fit(X_train_sv, y_train_sv)

    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, lr.predict(X_val))))
final_model = LogisticRegression(C=1)

final_model.fit(X_train, y_train)

print('Final Accuracy: {}'.format(accuracy_score(y_test, final_model.predict(X_test))))
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))

ngram_vectorizer.fit(train['Text'])



X_train = ngram_vectorizer.transform(train['Text'])

X_test = ngram_vectorizer.transform(test['Text'])

print(X_train.shape)



y_train = train['Sentiment'].values

y_test = test['Sentiment'].values



X_train_sv, X_val, y_train_sv, y_val = train_test_split(X_train, y_train, test_size=0.33)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    lr = LogisticRegression(C=c)

    lr.fit(X_train_sv, y_train_sv)

    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, lr.predict(X_val))))
final_model = LogisticRegression(C=1)

final_model.fit(X_train, y_train)

print('Final Accuracy: {}'.format(accuracy_score(y_test, final_model.predict(X_test))))
wc_vectorizer = CountVectorizer(binary=False)

wc_vectorizer.fit(train['Text'])



X_train = wc_vectorizer.transform(train['Text'])

X_test = wc_vectorizer.transform(test['Text'])

print(X_train.shape)



y_train = train['Sentiment'].values

y_test = test['Sentiment'].values



X_train_sv, X_val, y_train_sv, y_val = train_test_split(X_train, y_train, test_size=0.33)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    lr = LogisticRegression(C=c)

    lr.fit(X_train_sv, y_train_sv)

    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, lr.predict(X_val))))
final_model = LogisticRegression(C=0.5)

final_model.fit(X_train, y_train)

print('Final Accuracy: {}'.format(accuracy_score(y_test, final_model.predict(X_test))))
tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(train['Text'])



X_train = tfidf_vectorizer.transform(train['Text'])

X_test = tfidf_vectorizer.transform(test['Text'])

print(X_train.shape)



y_train = train['Sentiment'].values

y_test = test['Sentiment'].values



X_train_sv, X_val, y_train_sv, y_val = train_test_split(X_train, y_train, test_size=0.33)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    lr = LogisticRegression(C=c)

    lr.fit(X_train_sv, y_train_sv)

    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, lr.predict(X_val))))
final_model = LogisticRegression(C=1)

final_model.fit(X_train, y_train)

print('Final Accuracy: {}'.format(accuracy_score(y_test, final_model.predict(X_test))))
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))

ngram_vectorizer.fit(train['Text'])



X_train = ngram_vectorizer.transform(train['Text'])

X_test = ngram_vectorizer.transform(test['Text'])

print(X_train.shape)



y_train = train['Sentiment'].values

y_test = test['Sentiment'].values



X_train_sv, X_val, y_train_sv, y_val = train_test_split(X_train, y_train, test_size=0.33)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    svm = LinearSVC(C=c)

    svm.fit(X_train_sv, y_train_sv)

    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, svm.predict(X_val))))
final_model = LinearSVC(C=0.5)

final_model.fit(X_train, y_train)

print('Final Accuracy: {}'.format(accuracy_score(y_test, final_model.predict(X_test))))
additional_stop_words = ['in', 'of', 'at', 'a', 'the']

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=additional_stop_words)

ngram_vectorizer.fit(train['Text'])



X_train = ngram_vectorizer.transform(train['Text'])

X_test = ngram_vectorizer.transform(test['Text'])

print(X_train.shape)



y_train = train['Sentiment'].values

y_test = test['Sentiment'].values



X_train_sv, X_val, y_train_sv, y_val = train_test_split(X_train, y_train, test_size=0.33)



for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]:

    svm = LinearSVC(C=c)

    svm.fit(X_train_sv, y_train_sv)

    print('Accuracy for C={}: {}'.format(c, accuracy_score(y_val, svm.predict(X_val))))
final_model = LinearSVC(C=0.25)

final_model.fit(X_train, y_train)

print('Final Accuracy: {}'.format(accuracy_score(y_test, final_model.predict(X_test))))