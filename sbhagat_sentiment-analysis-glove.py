import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer #word stemmer class
lemma = WordNetLemmatizer()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
embeddings_index = dict()
f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
df_total = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')
display(df_total.head())
def preprocess_text(tweet):
    tweet = tweet.lower() # Convert to lowercase
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet) # Remove words with non-ASCII characters
    words = tweet.split()
    words = filter(lambda x: x[0]!= '@' , tweet.split()) # Remove user tags
    words = [word for word in words if word not in set(stopwords.words('english'))] # Remove stop words
    tweet = " ".join(words)
    return tweet
df_total['preprocessedTweet'] = df_total.tweet.apply(preprocess_text)
display(df_total.head())
df_total.isna().sum()
max_length = df_total.preprocessedTweet.apply(lambda x: len(x.split())).max()

t = Tokenizer()
t.fit_on_texts(df_total.preprocessedTweet)
vocab_size = len(t.word_index) + 1
encoded_tweets = t.texts_to_sequences(df_total.preprocessedTweet)
padded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding='post')

vocab_size = len(t.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
x_train, x_test, y_train, y_test = train_test_split(padded_tweets, df_total.label, test_size=0.2, stratify=df_total.label)
model_glove = Sequential()
model_glove.add(Embedding(vocab_size, 50, input_length=max_length, weights=[embedding_matrix], trainable=True))
model_glove.add(Bidirectional(LSTM(20, return_sequences=True)))
model_glove.add(Dropout(0.2))
model_glove.add(BatchNormalization())
model_glove.add(Bidirectional(LSTM(20, return_sequences=True)))
model_glove.add(Dropout(0.2))
model_glove.add(BatchNormalization())
model_glove.add(Bidirectional(LSTM(20)))
model_glove.add(Dropout(0.2))
model_glove.add(BatchNormalization())
model_glove.add(Dense(64, activation='relu'))
model_glove.add(Dense(64, activation='relu'))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit train data
model_glove.fit(x_train, y_train, epochs = 10)
y_pred = model_glove.predict(x_test)
pr, rc, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot(thresholds, pr[1:])
plt.plot(thresholds, rc[1:])
plt.show()
crossover_index = np.max(np.where(pr == rc))
crossover_cutoff = thresholds[crossover_index]
crossover_recall = rc[crossover_index]
print("Crossover at {0:.2f} with recall {1:.2f}".format(crossover_cutoff, crossover_recall))
print(classification_report(y_test, y_pred > crossover_cutoff))