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
from textblob import TextBlob

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import xgboost, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers





from warnings import filterwarnings

filterwarnings('ignore')
df = pd.read_csv("../input/train.tsv", sep = "\t")

df.sample(10)
df["Sentiment"].replace(0, value = "negative", inplace = True)

df["Sentiment"].replace(1, value = "negative", inplace = True)

df["Sentiment"].replace(3, value = "positive", inplace = True)

df["Sentiment"].replace(4, value = "positive", inplace = True)

df.sample(7)
df = df[df["Sentiment"]!=2].copy()
df.groupby("Sentiment").count()
df2 = pd.DataFrame()

df2["text"] = df["Phrase"]

df2["label"] = df["Sentiment"]

df = df2.copy() 

df.head()
df["text"] = df["text"].apply(lambda x:" ".join(x.lower() for x in x.split()))
df["text"] = df["text"].str.replace("[^\w\s]","")
df["text"] = df["text"].str.replace("\d","")
import nltk

from nltk.corpus import stopwords

sw = stopwords.words("english")

df["text"] = df["text"].apply(lambda x:" ".join(x for x in x.split() if x not in sw))
delete = pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]

df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in delete))
from textblob import Word

df["text"] = df["text"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
# count vectors

# tf idf vectors (words, characters, n-grams)

# word embeddings
df.head()
df.iloc[0]
train_x, test_x, train_y, test_y = model_selection.train_test_split(df["text"], df["label"])
encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

test_y = encoder.fit_transform(test_y)
#Count Vectors
vectorizer = CountVectorizer()

vectorizer.fit(train_x)
train_x_count = vectorizer.transform(train_x)

test_x_count = vectorizer.transform(test_x)
vectorizer.get_feature_names()[0:10]
train_x_count.toarray()
#tf - idf
tfidf_word_vectorizer = TfidfVectorizer()

tfidf_word_vectorizer.fit(train_x)
train_x_tfidf = tfidf_word_vectorizer.transform(train_x)

test_x_tfidf = tfidf_word_vectorizer.transform(test_x)
tfidf_word_vectorizer.get_feature_names()[:5]
train_x_tfidf.toarray()
#n-gram tf-idf
tfidf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))

tfidf_ngram_vectorizer.fit(train_x)
train_x_tfidf_ngram = tfidf_ngram_vectorizer.transform(train_x)

test_x_tfidf_ngram = tfidf_ngram_vectorizer.transform(test_x)
# characters level tf-idf
tfidf_chars_vectoizer = TfidfVectorizer(analyzer="char", ngram_range = (2,3))

tfidf_chars_vectoizer.fit(train_x)
train_x_tfidf_chars = tfidf_chars_vectoizer.transform(train_x)

test_x_tfidf_chars = tfidf_chars_vectoizer.transform(test_x)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(train_x_count, train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           test_x_count, 

                                           test_y, 

                                           cv = 10).mean()



print("Count Vectors Accuracy:", accuracy)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(train_x_tfidf,train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           test_x_tfidf, 

                                           test_y, 

                                           cv = 10).mean()



print("Word-Level TF-IDF Accuracy:", accuracy)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(train_x_tfidf_ngram,train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           test_x_tfidf_ngram, 

                                           test_y, 

                                           cv = 10).mean()



print("N-GRAM TF-IDF Accuracy:", accuracy)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(train_x_tfidf_chars,train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           test_x_tfidf_chars, 

                                           test_y, 

                                           cv = 10).mean()



print("CHARLEVEL Accuracy:", accuracy)