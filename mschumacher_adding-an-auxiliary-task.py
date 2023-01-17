import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm, tqdm_notebook
import math
from sklearn.model_selection import KFold
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, CuDNNLSTM, Conv1D, Add
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Lambda, Concatenate
from keras.optimizers import Nadam, Adam
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_n = len(train_df)
test_n = len(test_df)
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
# NLTK sentiment cell
print('\nGetting sentiments...')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
sentiments = np.zeros(train_n)
# for _, row in train_df.sample(10).iterrows():
#     print(row.question_text, sia.polarity_scores(row.question_text))

for i, (_, row) in tqdm_notebook(enumerate(train_df.iterrows()), total=train_n):
    sentiments[i] = sia.polarity_scores(row.question_text)['compound']

train_df['sentiment'] = pd.Series(sentiments)
train_df['sentiment_target'] = (train_df['sentiment'] + 1) / 2
# Get correlation between strong polarity and insincerity
print('\nCorrelations to polarity:')
# train_df['strong_polarity'] = pd.Series((train_df['sentiment'] >= 0.5) | (train_df['sentiment'] <= -0.5)).astype(int)
train_df['polarity'] = train_df['sentiment'].abs()
print('Pearson ', train_df['target'].corr(train_df['polarity'], method='pearson'))
print('Kendall ', train_df['target'].corr(train_df['polarity'], method='kendall'))
print('Spearman', train_df['target'].corr(train_df['polarity'], method='spearman'))
train_df.head()
print(train_df.describe())
print('\nPreparing data...')

# some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

# fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

# Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)

# Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

# Get the target values
train_y = train_df[['target', 'sentiment_target']].values
print('\nGetting embeddings...')
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in tqdm_notebook(word_index.items(), total=len(word_index)):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
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
    
    # So we only measure F1 on the target y value:
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
# Create a simple 1-layer LSTM model
def create_model(embedding_trainable=False, dropout=0.1, size=32, n_outputs=2, lr=0.003):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=embedding_trainable))
    model.add(SpatialDropout1D(dropout))
    model.add(Bidirectional(CuDNNLSTM(size, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation="sigmoid"))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=[f1]
    )

    return model
scores = []
for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=123).split(train_X, train_y):
    model = create_model(n_outputs=1)

    model.fit(
        train_X[train_idx], train_y[train_idx, 0],
        validation_data=(train_X[val_idx], train_y[val_idx, 0]),
        batch_size=1024,
        epochs=3,
        verbose=2,
    )

    scores.append(model.evaluate(train_X[val_idx], train_y[val_idx, 0], batch_size=1024, verbose=0)[1])
    print()

print('Average F1:', np.mean(scores))
print('Standard Deviation:', np.std(scores))
scores = []
for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=123).split(train_X, train_y):
    model = create_model(n_outputs=2)

    model.fit(
        train_X[train_idx], train_y[train_idx],
        validation_data=(train_X[val_idx], train_y[val_idx]),
        batch_size=1024,
        epochs=3,
        verbose=2,
    )

    scores.append(model.evaluate(train_X[val_idx], train_y[val_idx], batch_size=1024)[1])
    print()

print('Average F1:', np.mean(scores))
print('Standard Deviation:', np.std(scores))
scores = []
for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=123).split(train_X, train_y):
    model = create_model(n_outputs=1)

    model.fit(
        train_X[train_idx], train_y[train_idx, 1],
        validation_data=(train_X[val_idx], train_y[val_idx, 1]),
        batch_size=1024,
        epochs=2,
        verbose=2,
    )

    # Reduce the model's learning rate:
    model.compile(
        loss=model.loss,
        metrics=model.metrics,
        optimizer=Adam(lr=0.001),
    )
    
    # Fine-tuning:
    model.fit(
        train_X[train_idx], train_y[train_idx, 0],
        validation_data=(train_X[val_idx], train_y[val_idx, 0]),
        batch_size=512,
        epochs=2,
        verbose=2,
    )
    
    scores.append(model.evaluate(train_X[val_idx], train_y[val_idx, 0], batch_size=1024)[1])
    print()

print('Average F1:', np.mean(scores))
print('Standard Deviation:', np.std(scores))