import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import operator 
import re
import gc
import keras
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
train['target'].value_counts()
train.head()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_features = 50000
tk = Tokenizer(lower = True, filters='', num_words=max_features)
full_text = list(train['question_text'].values) + list(test['question_text'].values)
tk.fit_on_texts(full_text)

train_tokenized = tk.texts_to_sequences(train['question_text'].fillna('UNK'))
test_tokenized = tk.texts_to_sequences(test['question_text'].fillna('UNK'))
max_len = 70
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)
Y = train['target'].values
sub = test[['qid']]
path = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
emb_size = 300
def get_coef(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coef(*o.strip().split(" ")) for o in open(path, encoding='utf-8', errors='ignore'))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, emb_size))
for word, i in word_index.items():
    if i >= max_features: 
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector

del train_tokenized,test_tokenized,train,test
gc.collect()
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional,Input,Dropout,Flatten,Embedding,BatchNormalization
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
def lstm_model():
    keras.backend.clear_session()       
    inp = Input(shape=(70,))
    x = Embedding(max_features+1, emb_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)
    x = Flatten()(x)

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.12)(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
model = lstm_model()
model.summary()
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint

kfold = StratifiedKFold(n_splits=5,random_state=5, shuffle=True)
scores = []

for i, (train, valid) in enumerate(kfold.split(X_train, Y)):
    X_Train, X_val, Y_train, Y_val = X_train[train], X_train[valid], Y[train], Y[valid]
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks = [checkpoint]
    model = lstm_model()
    model.fit(X_Train, Y_train, batch_size=512, epochs=6, validation_data=(X_val, Y_val), verbose=2, callbacks=callbacks, )
    model.load_weights(filepath)
    y_pred = model.predict([X_val], batch_size=1024, verbose=2)
    y_pred = (y_pred>0.5) + 0
    
    score = f1_score(Y_val,y_pred)
    scores.append(score)
print(scores)
y_test = model.predict([X_test], batch_size=1024, verbose=2)
y_test = (y_test>0.5)+0
y_test = y_test.reshape((-1, 1))
sub['prediction']=y_test
sub.to_csv("submission.csv", index=False)
