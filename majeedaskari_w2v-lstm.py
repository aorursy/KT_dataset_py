from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.models import Sequential

from keras.layers import CuDNNLSTM, Dense, Bidirectional, Flatten, LSTM



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm, tqdm_notebook

import math

from sklearn.model_selection import train_test_split

import pickle

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import re

from gensim.models import Word2Vec



from nltk.stem import WordNetLemmatizer

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk.corpus import wordnet



lemmatizer = WordNetLemmatizer()

sw = nltk.corpus.stopwords.words('english')

sw.append(["?","’",",","?"])
# from google.colab import drive

# drive.mount('/content/drive')



train_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
print(len(train_df))

print(len(test_df))
def preprocess(text):

    # lower case:

    lower_text = text.lower()

    # tokenize lemmatize stopword removal:

    lemmatized_q = [lemmatizer.lemmatize(word=word,pos='v') for word in word_tokenize(lower_text.replace(",", " ").replace(".", " ").replace("?", " ").replace("-", " ").replace("\"", " ").replace("'", " ")) if word not in sw]

    # TODO: use phrases for better tokenization

    # TODO: use normalization for abbrv words ex: luv => love, fb => facebook,

    return lemmatized_q



lemmatized_qs = [preprocess(text) for text in tqdm_notebook(train_df['question_text'])]
lenmax = 40

i = 0

for x in lemmatized_qs:

    if len(x) > lenmax:

        i += 1

        

print(i/len(lemmatized_qs))



# if we choose 40 as lenght of sentence, only less than 0.0002 of data will be discarded, instead we will have a smaller faster network
w2v_model = Word2Vec(lemmatized_qs, size=300, window=5, min_count=1, workers=8)
def sentence_to_np_embedding(text):

    text = preprocess(text)[:40]

    embeddings = []

    for word in text:

      try:

        vector = w2v_model.wv[word]

      except Exception as e:

        vector = np.zeros(300)

      embeddings.append(vector)

    embeddings += [np.zeros(300)] * (40 - len(embeddings))

    return np.array(embeddings)
# Data providers

train_batch_size = 128

def train_batch_gen(train_df):

    n_batches = math.ceil(len(train_df) / train_batch_size)

    while True: 

        train_df = train_df.sample(frac=1.)  # Shuffle the data.

        for i in range(n_batches):

            texts = train_df.iloc[i*train_batch_size:(i+1)*train_batch_size, 1]

            text_arr = np.array([sentence_to_np_embedding(text) for text in texts])

            yield text_arr, np.array(train_df["target"][i*train_batch_size:(i+1)*train_batch_size])

            # yield text_arr, to_categorical(np.array(train_df["target"][i*train_batch_size:(i+1)*train_batch_size]), num_classes=None)
model = Sequential()

model.add(LSTM(128, input_shape=(40, 300)))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit_generator(train_batch_gen(train_df),

                    epochs=3,

                    steps_per_epoch=math.ceil(len(train_df) / train_batch_size),

                    verbose=True)
batch_size = 512

def test_batch_gen(test_df):

    n_batches = math.ceil(len(test_df) / batch_size)

    for i in range(n_batches):

        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]

        text_arr = np.array([sentence_to_np_embedding(text) for text in texts])

        yield text_arr



# test_df = pd.read_csv("drive/My Drive/Kaggle/test.csv")





all_preds = []

for x in tqdm_notebook(test_batch_gen(test_df)):

    all_preds.extend(model.predict(x).flatten())

# y_pred = []

# for i in all_preds:

#   if i > 0.100:

#     y_pred.append(1)

#   else:

#     y_pred.append(0)

    

y_te = (np.array(all_preds) > 0.400).astype(np.int)



submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

submit_df.to_csv("submission.csv", index=False)