import pandas as pd

import numpy as np

from keras.utils import np_utils

from sentiment_module import tokenize_stem



df = pd.read_csv("../input/sentiment-analysis-for-financial-news/all-data.csv", header = None, encoding='latin-1', names=["Sentiment", "Headlines"])

df['Sentiment'] = df['Sentiment'].replace("negative",0).replace("neutral",1).replace("positive",2)



corpus = []

for item in df['Headlines']:

    corpus.append(tokenize_stem(item))



from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()

y = df.iloc[:, 0].values
print(X.shape)

print(y.shape)
# transform column y to categorical data

y = np_utils.to_categorical(y, num_classes=3)
# Splitting into training sets and validation sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding

from keras.utils import np_utils



model = Sequential()

model.add(Dense(128, input_dim=(X_train.shape[1]), activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, y_train, epochs=20, batch_size=32)
model.evaluate(x=X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None)
from part1_cleaning import get_clean_data

df1, df2, df3 = get_clean_data()
from sentiment_module import tokenize_stem



# Predicting Headlines

corpus_hl1 = []

for item in df1['Headlines']:

    corpus_hl1.append(tokenize_stem(item))

pred_hl1 = cv.transform(corpus_hl1).toarray()

y_pred_hl1 = model.predict(pred_hl1)
print(y_pred_hl1.shape)

print(y_pred_hl1[0:10])
from sentiment_module import cluster_extraction



# Clustering Headlines

hl_sentiment = cluster_extraction(y_pred_hl1)

hl_sentiment[0:10]
# Predicting Descriptions/Previews

corpus_ds1 = []

for item in df1['Description']:

    corpus_ds1.append(tokenize_stem(item))

pred_ds1 = cv.transform(corpus_ds1).toarray()

y_pred_ds1 = model.predict(pred_ds1)
print(y_pred_ds1.shape)

print(y_pred_ds1[0:10])
# Clustering Descriptions/Previews

ds_sentiment = cluster_extraction(y_pred_ds1)

ds_sentiment[0:10]
from sentiment_module import combine_sentiments

ann_c_sentiment = combine_sentiments(hl_sentiment, ds_sentiment)

ann_c_sentiment[0:10]
# Headlines

corpus_hl2 = []

for item in df2['Headlines']:

    corpus_hl2.append(tokenize_stem(item))

pred_hl2 = cv.transform(corpus_hl2).toarray()

y_pred_hl2 = model.predict(pred_hl2)

print(y_pred_hl2.shape)
print(y_pred_hl2.shape)

print(y_pred_hl2[0:10])
# Clustering Headlines

hl_sentiment = cluster_extraction(y_pred_hl2)

hl_sentiment[0:10]
# Descriptions/Previews

corpus_ds2 = []

for item in df2['Description']:

    corpus_ds2.append(tokenize_stem(item))

pred_ds2 = cv.transform(corpus_ds2).toarray()

y_pred_ds2 = model.predict(pred_ds2)

print(y_pred_ds2.shape)
print(y_pred_ds2.shape)

print(y_pred_ds2[0:10])
# Clustering Descriptions/Previews

ds_sentiment = cluster_extraction(y_pred_ds2)

ds_sentiment[0:10]
from sentiment_module import combine_sentiments

ann_r_sentiment = combine_sentiments(hl_sentiment, ds_sentiment)

ann_r_sentiment[0:10]
# Headlines

corpus_hl3 = []

for item in df3['Headlines']:

    corpus_hl3.append(tokenize_stem(item))

pred_hl3 = cv.transform(corpus_hl3).toarray()

y_pred_hl3 = model.predict(pred_hl3)

print(y_pred_hl3.shape)
print(y_pred_hl3.shape)

print(y_pred_hl3[0:10])
# Clustering Headlines

hl_sentiment = cluster_extraction(y_pred_hl3)

hl_sentiment[0:10]
# The Guardian's headline sentiment is the only variavle dictate the sentiment of the Guardian's articles

ann_g_sentiment = hl_sentiment