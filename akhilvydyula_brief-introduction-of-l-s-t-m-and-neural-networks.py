import numpy as np

import pandas as pd

import os

from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from nltk.corpus import stopwords

import keras.preprocessing.text as T

from keras.preprocessing.text import Tokenizer

import nltk

nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
!ls ../input/quora-insincere-questions-classification
train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
x_train_all = train["question_text"].values

x_test = test["question_text"].values

y_train_all = train["target"].values



x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)



train_len = len(x_train)

val_len = len(x_val)

test_len = len(x_test)

print(train_len)

print(val_len)

print(test_len)
doc = np.concatenate((x_train, x_val, x_test))
vectorizer = TfidfVectorizer(stop_words='english', min_df=10, ngram_range=(1, 3))

# vectorizer = CountVectorizer(stop_words='english', min_df=5, ngram_range=(1, 3))

tfidf_model = vectorizer.fit(doc)

train_vector = tfidf_model.transform(x_train)

val_vector = tfidf_model.transform(x_val)

test_vector = tfidf_model.transform(x_test)
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(train_vector, y_train)
y_pred_val = mnb.predict(val_vector)

from sklearn.metrics import accuracy_score, precision_score, f1_score

acc_val = accuracy_score(y_val, y_pred_val)

pre_val = precision_score(y_val, y_pred_val)

f1_val = 2 / (1 / acc_val + 1 / pre_val)

print(acc_val, pre_val, f1_val)
y_pred_val_prob = mnb.predict_proba(val_vector)[:, 1]
y_pred_val_prob
from sklearn.metrics import f1_score

threshold_optimal = 0

f1_max = 0

divide_count = 100

for i in range(divide_count - 1):

    threshold = (i + 1) / divide_count

    y_pred_val = np.where(y_pred_val_prob > threshold, 1, 0)

    f1 = f1_score(y_val, y_pred_val)

    if f1 >= f1_max:

        threshold_optimal = threshold

        f1_max = f1

    # print(threshold, f1)

print(threshold_optimal, f1_max)
y_pred_test = np.where(mnb.predict_proba(test_vector)[:, 1] > threshold_optimal, 1, 0)

out_df = pd.DataFrame({"qid":test["qid"].values})

out_df['prediction'] = y_pred_test

out_df.to_csv("submission.csv", index=False)