from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import CuDNNLSTM,CuDNNGRU, Dense, Bidirectional, LSTM

train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

train_df, val_df = train_test_split(train_df, test_size=0.1)
#se intento con los otros encoders pero no supimos resolver el problema de data no compatible con el metodo.

embeddings_index = {}

f = open('../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')

for line in tqdm(f):

    values = line.split(" ")

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
def text_to_array(text):

    empyt_emb = np.zeros(300)

    text = text[:-1].split()[:30]

    embeds = [embeddings_index.get(x, empyt_emb) for x in text]

    embeds+= [empyt_emb] * (30 - len(embeds))

    return np.array(embeds)



val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])

val_y = np.array(val_df["target"][:3000])
# Data providers

batch_size = 128



def batch_gen(train_df):

    n_batches = math.ceil(len(train_df) / batch_size)

    while True: 

        train_df = train_df.sample(frac=1.)  # Shuffle the data.

        for i in range(n_batches):

            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]

            text_arr = np.array([text_to_array(text) for text in texts])

            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])
model = Sequential()

model.add(Bidirectional(LSTM(128, return_sequences=True),

                        input_shape=(30, 300)))

model.add(Bidirectional(LSTM(128)))

model.add(Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
mg = batch_gen(train_df)

model.fit_generator(mg, epochs=50,

                    steps_per_epoch=500,

                    validation_data=(val_vects, val_y),

                    verbose=True)
# prediction part

batch_size = 256

def batch_gen(test_df):

    n_batches = math.ceil(len(test_df) / batch_size)

    for i in range(n_batches):

        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]

        text_arr = np.array([text_to_array(text) for text in texts])

        yield text_arr



test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")



all_preds = []

for x in tqdm(batch_gen(test_df)):

    all_preds.extend(model.predict(x).flatten())
y_te = (np.array(all_preds) > 0.5).astype(np.int)



submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

submit_df.to_csv("submission.csv", index=False)