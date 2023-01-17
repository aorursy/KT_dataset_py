import pandas as pd

import matplotlib.pyplot as plt

import re

import string

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import model_selection

import math
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')
print(train_data.columns)
print("Shape(train_data): ", train_data.shape)

print("Shape(test_data): ",test_data.shape)
train_data.isnull().sum()
train_data['target'].value_counts()
plt.bar(train_data['target'].value_counts().index, train_data['target'].value_counts())

plt.ylabel("Counts")

plt.show()
print(train_data[train_data['target']==1]['text'][:10])
plt.barh(y=train_data['keyword'].value_counts()[:20].index, width=train_data['keyword'].value_counts()[:20])

plt.show()
def sanitize(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



train_data['text'] = train_data['text'].apply(lambda x: sanitize(x))
def remove_stopwords(words):

    res = []

    for w in words.split(' '):

        if w not in stopwords.words('english'):

            res.append(w)    

    return ' '.join(res)

train_data['text'] = train_data['text'].apply(lambda x: remove_stopwords(x))



train_data['text'].head()
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_data['text'])

test_vectors = count_vectorizer.transform(test_data["text"])
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train_data['text'])



model = MultinomialNB()

score = model_selection.cross_val_score(model, train_vectors, train_data["target"], cv=5, scoring="f1")

model.fit(train_vectors, train_data['target'])

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submission['target'] = model.predict(test_vectors)

submission.to_csv('submission.csv', index=False)