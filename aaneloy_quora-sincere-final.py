# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import os

import numpy as np 

import pandas as pd

#from tqdm.tqdm import tqdm

import math

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Dense,Activation, Flatten

from keras.layers import LSTM

from keras.layers import Dropout



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir())
train_df  = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")



print("Train Data Shape: ", train_df.shape)

print("Train Data Shape: ", test_df.shape)
train_df.head()
train_df, val_df = train_test_split(train_df, test_size=0.1)
EMBEDDING_FILE = "../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt"

embeddings_index = {} # Dictionary of word and its coefficients
f = open(EMBEDDING_FILE, encoding="utf8")

for line in f:

    values = line.split(" ")

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
val_df
# Convert values to embeddings

def text_to_array(text):

    empyt_emb = np.zeros(300)

    text = text[:-1].split()[:30]

    embeds = [embeddings_index.get(x, empyt_emb) for x in text]

    embeds+= [empyt_emb] * (30 - len(embeds))

    return np.array(embeds)



val_vects = np.array([text_to_array(X_text) for X_text in val_df["question_text"][:3000]])

val_y = np.array(val_df["target"][:3000])
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

# First Layer

model.add(LSTM(units=100, return_sequences=True, input_shape=(30, 300)))

model.add(Dropout(rate=0.2))



#2nd Layer

model.add(LSTM(units=100, return_sequences=True))

model.add(Dropout(rate=0.2))



#3rd Layer

model.add(LSTM(units=100, return_sequences=True))

model.add(Dropout(rate=0.2))



#4th Layer

model.add(LSTM(units=100, return_sequences=True))

model.add(Dropout(rate=0.2))



model.add(Flatten())



#Output Layes

model.add(Dense(units=1, activation="sigmoid"))



# Compile RNN

model.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['accuracy'])
mg = batch_gen(train_df)

model.fit_generator(mg, epochs=20,steps_per_epoch=1000,validation_data=(val_vects, val_y),verbose=True)
model.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 256

def batch_gen(test_df):

    n_batches = math.ceil(len(test_df) / batch_size)

    for i in range(n_batches):

        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]

        text_arr = np.array([text_to_array(text) for text in texts])

        yield text_arr



all_preds = []

for x in batch_gen(test_df):

    all_preds.extend(model.predict(x).flatten())
y_test = (np.array(all_preds) > 0.5).astype(np.int)



submit = pd.DataFrame({"qid": test_df["qid"], "prediction": y_test})

submit.to_csv("submission.csv", index=False)  #prediction of test data