import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

# text preprocessing
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# deep learning model
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.layers import Embedding, Input, Dense, LSTM, GlobalMaxPooling1D, GRU, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

# regularizers
from keras.regularizers import L1L2, l2
# Wordcloud
from wordcloud import WordCloud

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")
df.shape
df.describe()
df.info()
df.sample(5)
df[["text", "airline_sentiment"]]
def remove_stopwords(text):
    "removes stopwords"
    # customize stopwords
    my_stopwords = stopwords.words("english")
    my_stopwords.remove("not")
    # remove stopwords
    filtered_words = [word for word in text if word not in my_stopwords]
    return filtered_words

def remove_punctuation(text):
    "removes punctutation"
    my_punct = string.punctuation
    my_punct += "“”’" # add unusual apostrophes
    no_punct = [w for w in text if w not in my_punct]
    return no_punct

def remove_numbers(text):
    "removes strings containing only digits"
    reduced = re.sub(r'\b[0-9]+\b\s*', '', text)
    return reduced

def remove_signs(text, sign):
    "removes a particular sign"
    try:
        reduced = [w for w in text if sign not in w]
        return reduced
    except Exception as e:
        print(e)
        return text
    
def remove_links(text):
    "removes links"
    reduced = re.sub(r'http\S+', '', text)
    return reduced

def clean_text(inp_text):
    """
        This function is a pipeline for text preprocessing
        It consists of following steps:
            - converting text to lowercase
            - removing words containing only digits
            - removing links
            - removing stopwords
            - removing punctuation
            - removing mentions and hashtags
    """
    #lowercase
    text = inp_text.lower()
    # remove only number words
    text = remove_numbers(text)
    # remove links 
    text = remove_links(text)
    # divide input sentence into words
    tknzr = TweetTokenizer()
    text = tknzr.tokenize(text)
    # remove stopwords
    text = remove_stopwords(text)
    # remove punctuation
    text = remove_punctuation(text)
    # remove mentions
    text = remove_signs(text, "@")
    # remove hashtags
    text = remove_signs(text, "#")
    # join a list of words into a sentence
    filtered_sentence = (" ").join(text)
    return filtered_sentence
df["filtered_tweets"] = df["text"].apply(clean_text)
neg = " ".join(tweet for tweet in df[df["airline_sentiment"]=="negative"]["filtered_tweets"])
neu = " ".join(tweet for tweet in df[df["airline_sentiment"]=="neutral"]["filtered_tweets"])
pos = " ".join(tweet for tweet in df[df["airline_sentiment"]=="positive"]["filtered_tweets"])
plt.figure(figsize=(12, 9))
neg_cloud = WordCloud(max_words=50).generate(neg)
plt.imshow(neg_cloud, interpolation='bilinear')
plt.axis("off");
plt.figure(figsize=(12, 9))
pos_cloud = WordCloud(max_words=50).generate(pos)
plt.imshow(pos_cloud, interpolation='bilinear')
plt.axis("off");
plt.figure(figsize=(12, 9))
neu_cloud = WordCloud(max_words=50).generate(neu)
plt.imshow(neu_cloud, interpolation='bilinear')
plt.axis("off");
def get_labels(sentiment):
    if sentiment == "negative":
        return 0
    if sentiment == "neutral":
        return 1
    if sentiment == "positive":
        return 2
    
df["label"] = df["airline_sentiment"].apply(get_labels)
df["airline_sentiment"].hist();
# labels distribution
df["label"].value_counts() / df["label"].value_counts().sum()
train_txt, test_txt, train_labels, test_labels = train_test_split(
    df["filtered_tweets"], df["label"], 
    test_size=0.2, stratify=df["label"], random_state=42)
train_labels.value_counts() / train_labels.value_counts().sum()
tknz = Tokenizer()
tknz.fit_on_texts(train_txt)
train_sentences = tknz.texts_to_sequences(train_txt)
test_sentences = tknz.texts_to_sequences(test_txt)
vocab_size = len(tknz.word_counts)
vocab_size
# add padding to the train set
train_pad = pad_sequences(train_sentences)

# max length of words in a single sentence
max_len = train_pad.shape[1]

# add padding to the test set using max len
test_pad = pad_sequences(test_sentences, maxlen=max_len)
train_pad.shape, test_pad.shape
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
y_train[:4]
# size of the Embeddings vector
D = 15
# size of the hidden vector
M = 10
# vocab size
V = vocab_size
# sequence length
T = max_len

batch_size = 32

i = Input(shape=(T,))
x = Embedding(V+1, D)(i)
x = LSTM(M)(x)
x = Dense(3, activation='softmax')(x)

mbasic = Model(i, x)
# model.summary()

mbasic.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

basic = mbasic.fit(train_pad, y_train, batch_size=batch_size, epochs=10, validation_data=(test_pad, y_test))
# size of the Embeddings vector
D = 15
# size of the hidden vector
M = 64
# vocab size
V = vocab_size
# sequence length
T = max_len

batch_size = 32

i = Input(shape=(T,))
x = Embedding(V+1, D)(i)
x = LSTM(M)(x)
x = Dense(3, activation='softmax')(x)

mbig = Model(i, x)
# model.summary()

mbig.compile(optimizer=Adam(lr=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

big = mbig.fit(train_pad, y_train, batch_size=batch_size, epochs=10, validation_data=(test_pad, y_test))
# size of the Embeddings vector
D = 10
# size of the hidden vector
M = 10
# vocab size
V = vocab_size
# sequence length
T = max_len

batch_size = 32

i = Input(shape=(T,))
x = Embedding(V+1, D)(i)
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(3, activation='softmax')(x)

pool = Model(i, x)
# model.summary()

pool.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

lstm_pool = pool.fit(train_pad, y_train, batch_size=batch_size, epochs=10, validation_data=(test_pad, y_test))
# size of the Embeddings vector
D = 15
# size of the hidden vector
M = 10
# vocab size
V = vocab_size
# sequence length
T = max_len

batch_size = 32

i = Input(shape=(T,))
x = Embedding(V+1, D)(i)
x = LSTM(M, recurrent_dropout=0.2)(x)
x = Dense(3, activation='softmax')(x)

rec_drop = Model(i, x)
# model.summary()

rec_drop.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

rd = rec_drop.fit(train_pad, y_train, batch_size=batch_size, epochs=10, validation_data=(test_pad, y_test))
import keras.backend as K
from keras.layers import Lambda
# size of the Embeddings vector
D = 15
# size of the hidden vector
M = 10
# vocab size
V = vocab_size
# sequence length
T = max_len

batch_size = 32

i = Input(shape=(T,))
mask = Lambda(lambda inputs: K.not_equal(inputs, 0))(i)
x = Embedding(V+1, D)(i)
x = LSTM(M, return_sequences=True)(x, mask=mask)
x = LSTM(M)(x, mask=mask)
# x = GlobalMaxPooling1D()(x)
x = Dense(3, activation='softmax')(x)

mmask = Model(i, x)
# model.summary()

mmask.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

mask = mmask.fit(train_pad, y_train, batch_size=batch_size, epochs=10, validation_data=(test_pad, y_test))
# size of the Embeddings vector
D = 15
# size of the hidden vector
M = 10
# vocab size
V = vocab_size
# sequence length
T = max_len

batch_size = 64

i = Input(shape=(T,))
x = Embedding(V+1, D)(i)
x = GRU(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(3, activation='softmax')(x)

mgru = Model(i, x)
# model.summary()

mgru.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

gru = mgru.fit(train_pad, y_train, epochs=10, batch_size=batch_size, validation_data=(test_pad, y_test))
from keras.layers import BatchNormalization, Conv1D, Flatten
# size of the Embeddings vector
D = 15
# size of the hidden vector
M = 10
# vocab size
V = vocab_size
# sequence length
T = max_len

batch_size = 64

i = Input(shape=(T,))
x = Embedding(V+1, D)(i)
x = Conv1D(32, kernel_size=5)(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
# x = Conv1D(32, kernel_size=3)(x)
# x = Dropout(0.2)(x)
# x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(3, activation='softmax')(x)

mcnn = Model(i, x)
# model.summary()

mcnn.compile(optimizer=Adam(lr=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

cnn = mcnn.fit(train_pad, y_train, batch_size=batch_size, epochs=10, validation_data=(test_pad, y_test))
plt.plot(cnn.history['val_loss'], label="cnn val loss")
plt.plot(basic.history['val_loss'], label="Basic LSTM val loss")
plt.plot(rd.history['val_loss'], label="rec_dr val loss")
plt.plot(lstm_pool.history['val_loss'], label="LSTM with pool val loss")
plt.plot(gru.history['val_loss'], label="GRU val loss")
plt.legend();
plt.plot(cnn.history['val_accuracy'], label="cnn val acc")
plt.plot(basic.history['val_accuracy'], label="Basic LSTM val acc")
plt.plot(rd.history['val_accuracy'], label="rec_dr val acc")
plt.plot(lstm_pool.history['val_accuracy'], label="LSTM with pool val acc")
plt.plot(gru.history['val_accuracy'], label="GRU val acc")
plt.legend();
from sklearn.metrics import confusion_matrix

pred = rec_drop.predict(test_pad)
y_classes = pred.argmax(axis=-1)
y_classes
test_labels.values
confusion_matrix(y_classes, test_labels)
