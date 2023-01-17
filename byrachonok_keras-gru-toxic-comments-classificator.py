import collections
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import GRU, SpatialDropout1D
from keras import metrics
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_dataset = pd.read_csv('../input/train.csv').fillna(' ')
test_dataset = pd.read_csv('../input/test.csv').fillna(' ')
corpus = pd.concat([train_dataset['comment_text']])
%%time
MAX_FEATURE = 5000
MAX_LEN = 80

tk = Tokenizer(num_words = MAX_FEATURE, lower = True)
tk.fit_on_texts(corpus.str.lower())

X = tk.texts_to_sequences(train_dataset['comment_text'].str.lower())
X = pad_sequences(X, maxlen=MAX_LEN)
y = train_dataset[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
def model_create():
    model = Sequential()
    model.add(Embedding(MAX_FEATURE, MAX_LEN))
    model.add(SpatialDropout1D(0.3))
    model.add(GRU(100, dropout=0.3, recurrent_dropout=0.25)) 
    model.add(Dense(6, activation="sigmoid"))
    return model

def model_compile(model):
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['acc'])
    return model

def model_fit(model, X, y, batch_size=32, epochs=1, validation_split=0.2):
    history = model.fit(
        X, y, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_split=validation_split, 
        verbose=1)
    return history
model = model_create()
model_compile(model)
model_fit(model, X, y, batch_size=128, epochs=5)
model_fit(model, X, y, batch_size=256, epochs=5)
model_fit(model, X, y, batch_size=512, epochs=5)
model_fit(model, X, y, batch_size=1024, epochs=5)
model_fit(model, X, y, batch_size=2048, epochs=10)
model_fit(model, X, y, batch_size=4096, epochs=15)
predict = model.predict(
    pad_sequences(tk.texts_to_sequences(test_dataset['comment_text'].str.lower()), maxlen=MAX_LEN),verbose=1)
submission = pd.read_csv('../input/sample_submission.csv')
submission[class_names] = (predict)
submission.to_csv("submission.csv", index = False)