import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.feature_extraction.text import CountVectorizer

import re
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
import nltk

from nltk.corpus import stopwords

stop_set = set(stopwords.words('english'))
def remove_stop_words(texts):

    new_texts = []

    for i, text in enumerate(texts):

        T = []

        text = re.sub(r'[^\w\s]', ' ', text)

        text = re.sub(r'[\s]{2,}', ' ', text)

        text = text.strip()

        for t in text.split(' '):

            if t not in stop_set:

                T += [t]

        new_texts += [' '.join(T)]

    return new_texts
train_df.insert(5, 'new_text', remove_stop_words(train_df['text']))

test_df.insert(4, 'new_text', remove_stop_words(test_df['text']))
train_df
test_df
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

train_vectors = vectorizer.fit_transform(train_df['new_text'])

test_vectors = vectorizer.transform(test_df['new_text'])

clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
