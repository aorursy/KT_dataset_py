from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import os

import pandas as pd

import numpy as np

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("../input/train.csv")

train_df, val_df = train_test_split(train_df,test_size = 0.07)
train_df.question_text.str.split().str.len().describe()
SEQ_LEN = 100 # we set max length of each to be 100 words
embeddings_index = {}

f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')



for line in tqdm(f):

    values = line.split(" ")

    word = values[0]

    coefs = np.asarray(values[1:],dtype = 'float32')

    embeddings_index[word] = coefs

f.close()
print("Lenth of vector is ",len(embeddings_index['speech']),"\n","Vector for word speech","\n",embeddings_index['speech'])
len(embeddings_index)
import re

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

_DIGIT_RE = re.compile(br"\d")

STOP_WORDS = "\" \' [ ] . , ! : ; ?".split(" ")

def basic_tokenizer(sentence):

    """Very basic tokenizer: split the sentence into a list of tokens."""

    words = []

    for space_separated_fragment in sentence.strip().split():

        words.extend(_WORD_SPLIT.split(space_separated_fragment))

        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']

    return [w.lower() for w in words if w != '' and w != ' ']
def text_to_array(text):

    empyt_emb = np.zeros(300)

    text = basic_tokenizer(text[:-1])[:SEQ_LEN]

    embeds = [embeddings_index.get(x, empyt_emb) for x in text]

    embeds+= [empyt_emb] * (SEQ_LEN - len(embeds))

    return np.array(embeds)
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])

val_y = np.array(val_df["target"][:3000])
batch_size = 256



def batch_gen(train_df):

    n_batches = math.ceil(len(train_df) / batch_size)

    while True: 

        train_df = train_df.sample(frac=1.)  # Shuffle the data.

        for i in range(n_batches):

            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]

            text_arr = np.array([text_to_array(text) for text in texts])

            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])
from keras.models import Sequential,Model

from keras.layers import CuDNNLSTM, Dense, Bidirectional,Input,Dropout



from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers,regularizers, constraints
model = Sequential()

model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True),input_shape = (SEQ_LEN,300)))

model.add(Bidirectional(CuDNNLSTM(64)))

model.add(Dense(256,activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
mg = batch_gen(train_df)

model.fit_generator(mg, epochs=5,

                    steps_per_epoch=1000,

                    validation_data=(val_vects, val_y),

                    verbose=True)
batch_size = 256

def batch_gen(test_df):

    n_batches = math.ceil(len(test_df) / batch_size)

    for i in range(n_batches):

        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]

        text_arr = np.array([text_to_array(text) for text in texts])

        yield text_arr



test_df = pd.read_csv("../input/test.csv")



all_preds = []

for x in tqdm(batch_gen(test_df)):

    all_preds.extend(model.predict(x).flatten())
y_te = (np.array(all_preds) > 0.5).astype(np.int)



submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

submit_df.to_csv("submission.csv", index=False)
model_json = model.to_json()
with open("model_questionS.json", "w") as json_file:

    json_file.write(model_json)
model.save_weights("model_questionS.h5")