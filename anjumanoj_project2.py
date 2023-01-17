# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



!ls ../input/embeddings/
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers.embeddings import Embedding

## Plotly
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from nltk.stem import SnowballStemmer
from string import punctuation
from plotly import tools
import seaborn as sns


import tensorflow as tf
import numpy as np

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.utils import shuffle

import random


# fix random seed for reproducibility
np.random.seed(7)
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print("Train dataset shape : ", train_data.shape)
print("Test dataset shape : ", test_data.shape)
train_data.head()
train_data.tail()
train_data.info()
train_data.describe()
train_data['target'].value_counts()
train0_df0 = train_data[train_data["target"]==0]
train0_df0
train1_df1 = train_data[train_data["target"]==1]
train1_df1
def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

get_num_words_per_sample(train_data['question_text'])
get_num_words_per_sample(test_data['question_text'])
def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.
    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts],50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
plt.show()

plot_sample_length_distribution(train_data['question_text'])

plot_sample_length_distribution(test_data['question_text'])
digit_train, counts_train = np.unique(train_data['target'], return_counts = True)

distribution_train = dict(zip(digit_train, counts_train))
print(distribution_train )

plt.bar(list(distribution_train.keys()),distribution_train.values(),width =0.6)
plt.xlabel('Target --> 0(Sincere),1(Insincere))')
plt.ylabel('Number of Questions')
plt.show()
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS

train1_df = train_data[train_data["target"]==1]
train0_df = train_data[train_data["target"]==0]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS ]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')

#plt.figure(figsize=(10,16))
#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")
#plt.title("Frequent words for Insincere Questions", fontsize=16)
#plt.show()
train_data = shuffle(train_data)

from sklearn.model_selection import train_test_split

sentences = train_data['question_text'].values
target = train_data['target'].values

X_Train, X_Val, Y_Train,Y_Val = train_test_split(sentences, target, test_size=0.20, random_state=7)
shuffle_index = np.random.permutation(1044897)
X_Train, Y_Train = X_Train[shuffle_index], Y_Train[shuffle_index]
shuffle_index = np.random.permutation(261225)
X_Val, Y_Val = X_Val[shuffle_index], Y_Val[shuffle_index]
print(X_Train.size)
print(Y_Train.size)
print(X_Val.size)
print(Y_Val.size)
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 100


def sequence_vectorize(train_texts, val_texts,test_texts):
    """Vectorizes texts as sequence vectors.
    1 text = 1 sequence vector withfixed length.
    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.
        test_texts: list, validation text strings.
    # Returns
        x_train, x_val,x_test,tokenizer object, word_index: vectorized training and validation and test
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    #num_words, which is responsible for setting the size of the vocabulary.
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)
    x_test = tokenizer.texts_to_sequences(test_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_length)
    return x_train, x_val,x_test,tokenizer.word_index,tokenizer


x_train ,x_val ,x_test,word_index,tk = sequence_vectorize(X_Train,X_Val,test_data['question_text'].values)
x_train


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def load_embedding(file):
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
    return embeddings_index


def make_embedding_matrix(embedding, tokenizer, len_voc):
    all_embs = np.stack(embedding.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = tokenizer.word_index
    embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))
    
    for word, i in word_index.items():
        if i >= len_voc:
            continue
        embedding_vector = embedding.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    
    return embed_size,embedding_matrix
glove = load_embedding('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')

embedding_dim,embed_mat = make_embedding_matrix(glove, tk, TOP_K)
print(embedding_dim)
# Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
num_features = min(len(word_index) + 1, TOP_K)
print(num_features)


model4= Sequential()
model4.add(Embedding(input_dim=num_features,output_dim=embedding_dim,input_length=100, weights=[embed_mat],trainable=True))
model4.add(Dropout(0.2))
model4.add(Conv1D(64, 5, activation='relu'))
model4.add(MaxPooling1D(pool_size=4))
model4.add(LSTM(100))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model4.summary()  
history = model4.fit(x_train, Y_Train,
                    epochs=10,
                    verbose=True,
                    validation_split=0.2,
                    batch_size=1024)
loss, accuracy = model4.evaluate(x_val, Y_Val, verbose=True)
print("Test Accuracy: {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    


plot_history(history)
model5= Sequential()
model5.add(Embedding(input_dim=num_features,output_dim=embedding_dim,input_length=100, weights=[embed_mat],trainable=True))
model5.add(Dropout(0.2))
model5.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model5.add(Dense(1, activation='sigmoid'))
model5.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model5.summary()  
history5 = model5.fit(x_train, Y_Train,
                    epochs=10,
                    verbose=True,
                    validation_split=0.2,
                    batch_size=1024)
loss, accuracy = model5.evaluate(x_val, Y_Val, verbose=True)
print("Test Accuracy: {:.4f}".format(accuracy))
plot_history(history5)
y_predict = model5.predict(x_test, batch_size=None, verbose=1, steps=None)
test_df = pd.read_csv("../input/test.csv")
y_predict
y_pred_changed = y_predict.argmax(1)
y_pred_changed
y_te = (np.array(y_predict) > 0.5).astype(np.int)
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te.flatten()})
submit_df.to_csv("submission.csv", index=False)
y_predicted = model5.predict(x_val, batch_size=None, verbose=1, steps=None)
y_predicted_changed = y_predicted.argmax(1)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Val, y_pred_changed)
cm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
accuracy_score(sub['prediction'], y_pred_changed)
f1_score(sub['prediction'], y_pred_changed,average='weighted')
recall_score(sub['prediction'], y_pred_changed,average='weighted')
precision_score(sub['prediction'], y_pred_changed,average='weighted')
