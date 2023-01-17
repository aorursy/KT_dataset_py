import os
import time
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D,MaxPooling1D,BatchNormalization
from keras.models import Model
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train, dev = train_test_split(train, test_size=0.1, random_state=2018)

embed_size = 300 
max_features = 50000 
maxlen = 60 

train_X = train["question_text"].fillna("_na_").values
dev_X = dev["question_text"].fillna("_na_").values
test_X = test["question_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
dev_X = tokenizer.texts_to_sequences(dev_X)
test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=maxlen)
dev_X = pad_sequences(dev_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

train_y = train['target'].values
dev_y = dev['target'].values
model=None
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
from keras import backend as K

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
def net(input_shape):
    sentence_indices = Input(input_shape, dtype='int32')
    X = Embedding(max_features, embed_size,weights=[embedding_matrix])(sentence_indices)
    X = Bidirectional(CuDNNGRU(64, return_sequences=True))(X)
    X = GlobalMaxPool1D()(X)
    X = Dense(16, activation="relu")(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation="sigmoid")(X)
    model = Model(inputs=sentence_indices, outputs=X)
    return model
model=net((maxlen,))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1])

print(model.summary())
weight={0:0.4,1:1}

model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(dev_X, dev_y),class_weight=weight)
pred_dev_y = model.predict([dev_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.601, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(dev_y, (pred_dev_y>thresh).astype(int))))
label = model.predict([test_X], batch_size=1024, verbose=1)
label=(label>0.58).astype(int)
submission=pd.read_csv("../input/sample_submission.csv")
submission["prediction"]=label
submission.head()