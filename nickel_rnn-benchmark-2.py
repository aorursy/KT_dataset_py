import gc

import numpy as np 

import pandas as pd

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, CuDNNLSTM

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping

plt.style.use('fivethirtyeight') 

%matplotlib inline
data = pd.concat([

       pd.read_csv("../input/pageviews/pageviews.csv", parse_dates=["FEC_EVENT"]),

       pd.read_csv("../input/pageviews_complemento/pageviews_complemento.csv", parse_dates=["FEC_EVENT"])

])
y_prev = pd.read_csv("../input/conversiones/conversiones.csv")

y_train = pd.Series(0, index=sorted(data.USER_ID.unique()))

y_train.loc[y_prev[y_prev.mes >= 10].USER_ID.unique()] = 1
pages = data[data.FEC_EVENT.dt.month < 10].groupby("PAGE").USER_ID.unique()

pages = pages.index[pages.apply(lambda x: y_train.loc[x].mean() / y_train.mean() - 1).abs() > 0.05]

pages_set = set(pages)

pages = data.PAGE.value_counts()

pages = pages.index[(pages > 10) & (pages < pages.iloc[int(pages.shape[0] * 0.1)])]

pages_set = list(pages_set.intersection(pages))
history = 500



train = data.groupby("USER_ID").apply(lambda x: x[(x.FEC_EVENT.dt.month < 10) & x.PAGE.isin(pages_set)]\

                                      .sort_values("FEC_EVENT").PAGE[-history:].values)

train = pad_sequences(train)

train
import tensorflow as tf

from sklearn.metrics import roc_auc_score



def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def getModel():

    model = Sequential()

    model.add(Embedding(train.max() + 1, 32, input_length=history))

    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))

    model.add(Dropout(0.2))

    model.add(Bidirectional(CuDNNLSTM(32)))

    model.add(Dropout(0.2))

    model.add(Dense(64, activation="relu"))

    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))

    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auroc])

    return model
model = getModel()

model.summary()
model.fit(train, y_train, batch_size=2500, epochs=1000, verbose=1, validation_split=0.1,

          callbacks=[EarlyStopping(monitor='val_auroc', patience=10, verbose=1, mode='max', restore_best_weights=True)])
test = data.groupby("USER_ID").apply(lambda x: x[x.PAGE.isin(pages_set)].sort_values("FEC_EVENT").PAGE[-history:].values)

test = pad_sequences(test)
test_probs = []

for i in range(10):

    model = getModel()

    model.fit(train, y_train, batch_size=1000, epochs=1000, verbose=1,

                validation_split=0.1,

                callbacks=[EarlyStopping(monitor='val_auroc', patience=10, verbose=1,

                                         mode="max", restore_best_weights=True)])

    

    test_probs.append(pd.Series(model.predict(test)[:, -1], name="fold_" + str(i)))



test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

test_probs.index.name="USER_ID"

test_probs.name="SCORE"



test_probs.to_csv("rnn2_benchmark.zip", header=True, compression="zip")