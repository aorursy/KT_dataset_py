# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.shape
test.head()
#df = df.reindex(np.random.permutation(df.index))  
train = train[['question_text', 'target']]
test_X = test[['question_text']]
train.head()
test_X.head()
sns.countplot(x='target', data=train)
NB_WORDS = 50000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000  # Size of the validation set
EPOCHS = 5  # Number of epochs we usually start to train with
BATCH_SIZE = 1024  # Size of the batches used in the mini-batch gradient descent
stopwords_list = list(STOP_WORDS)
def remove_stopwords(input_text):
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
train.question_text = train.question_text.apply(remove_stopwords)
test_X.question_text = test_X.question_text.apply(remove_stopwords)
X_train, X_valid, y_train, y_valid = train_test_split(train.question_text, train.target, test_size=0.1, 
                                                    random_state=37, stratify = train.target)
print('Training Data:', X_train.shape[0])
print('Validation Data:', X_valid.shape[0])
tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
tk.fit_on_texts(X_train)
print('Top 5 most common words are:', collections.Counter(tk.word_counts).most_common(5))
word_index = tk.word_index
print('Found %s unique tokens.' % len(word_index))
X_train_seq = tk.texts_to_sequences(X_train)
X_valid_seq = tk.texts_to_sequences(X_valid)
X_test_seq = tk.texts_to_sequences(test_X.question_text)
len(X_test_seq)
seq_lengths = X_train.apply(lambda x: len(x.split(' ')))
seq_lengths.describe()
print('{} -- is converted to -- {}'.format(X_train[7], X_train_seq[7]))
MAX_LEN = 50
X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_valid_seq_trunc = pad_sequences(X_valid_seq, maxlen=MAX_LEN)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN)
print('{} -- is converted to -- {}'.format(X_train_seq[7], X_train_seq_trunc[7]))
print('Shape of train set:',X_train_seq_trunc.shape)
print('Shape of validation set:',X_valid_seq_trunc.shape)
print('Shape of test set:',X_test_seq_trunc.shape)
# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
model = models.Sequential()
model.add(layers.Embedding(NB_WORDS, 8, input_length=MAX_LEN))
model.add(layers.CuDNNLSTM(64))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
#history = model.fit(X_train_seq_trunc, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    #validation_data=(X_valid_seq_trunc, y_valid))
#pred_noemb_y = model.predict([X_test_seq_trunc], batch_size=1024, verbose=1)
#pred_noemb_y = (pred_noemb_y>0.35).astype(int)
#pred_noemb_y[0:5]
#out_df = pd.DataFrame({"qid":test["qid"].values})
#out_df['prediction'] = pred_noemb_y
#out_df.to_csv("submission.csv", index=False)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tk.word_index
nb_words = min(NB_WORDS, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
glove_model = models.Sequential()
glove_model.add(layers.Embedding(NB_WORDS, embed_size, input_length=MAX_LEN, 
                                 weights=[embedding_matrix],trainable=False))
glove_model.add(layers.CuDNNLSTM(64))
glove_model.add(layers.Dropout(0.2))
glove_model.add(layers.Dense(16, activation='relu'))
glove_model.add(layers.Dropout(0.2))
glove_model.add(layers.Dense(1, activation='sigmoid'))
glove_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
glove_model.summary()
history = glove_model.fit(X_train_seq_trunc, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    validation_data=(X_valid_seq_trunc, y_valid))
pred_emb_y = glove_model.predict([X_test_seq_trunc], batch_size=1024, verbose=1)
pred_emb_y = (pred_emb_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test["qid"].values})
out_df['prediction'] = pred_emb_y
out_df.to_csv("submission.csv", index=False)
