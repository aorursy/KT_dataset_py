# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_train.csv', encoding='latin1')
test = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_test.csv', encoding='latin1')
train
test
le = LabelEncoder()
X_train = train['OriginalTweet']
y_train = le.fit_transform(train['Sentiment'])

X_test = test['OriginalTweet']
y_test = le.transform(test['Sentiment'])
BoW = CountVectorizer()
tfidf = TfidfTransformer()
tok = Tokenizer(num_words=50, split=" ")
X_tr_bow = BoW.fit_transform(X_train)
X_te_bow = BoW.transform(X_test)

X_tr_tfidf = tfidf.fit_transform(X_tr_bow)
X_te_tfidf = tfidf.transform(X_te_bow)

tok.fit_on_texts(X_train)
tok_train = tok.texts_to_sequences(X_train)
X_tr_tok = sequence.pad_sequences(tok_train, maxlen=11, dtype='float32')

tok_test = tok.texts_to_sequences(X_test)
X_te_tok = sequence.pad_sequences(tok_test, maxlen=11, dtype='float32')
wordcloud = WordCloud(background_color='white').generate(" ".join(X))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
month = pd.to_datetime(train['TweetAt']).dt.month
day = pd.to_datetime(train['TweetAt']).dt.day

count = Counter(day)
plt.bar(count.keys(), count.values(), color='blue')
plt.xlabel('Days')
plt.ylabel('Occurence')
plt.title('Distribution of the tweets per day')
plt.show()

count = Counter(month)
plt.bar(count.keys(), count.values(), color='red')
plt.xlabel('Month')
plt.ylabel('Occurence')
plt.title('Distribution of the tweets per month')
plt.show()
bow_clf = SGDClassifier(eta0=0.01, learning_rate='optimal', penalty='l1', max_iter=100)
bow_clf.fit(X_tr_bow, y_train)
bow_score = bow_clf.score(X_te_bow, y_test)
print('Bag of Words score: ' + str(bow_score))

tfidf_clf = SGDClassifier(eta0=0.0001, loss='modified_huber', penalty='l1', learning_rate='optimal')
tfidf_clf.fit(X_tr_tfidf, y_train)
tfidf_score = tfidf_clf.score(X_te_tfidf, y_test)
print('TFIDF score:        ' + str(tfidf_score))

tok_clf = SGDClassifier()
tok_clf.fit(X_tr_tok, y_train)
tok_score = tok_clf.score(X_te_tok, y_test)
print('Tokenizer score:    ' + str(tok_score))
pred = bow_clf.predict(X_te_bow)
output = pd.DataFrame({'Real': y_test, 'Prediction': pred})
output.to_csv('submission.csv', index=False)