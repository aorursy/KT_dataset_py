# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# nltk

import nltk

from nltk.corpus import stopwords

from  nltk.stem import SnowballStemmer

import re



# Scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, CuDNNLSTM, CuDNNGRU, Bidirectional

from keras import utils

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# !wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
# !unzip glove.twitter.27B.zip glove.twitter.27B.100d.txt
# !ls
# DATASET

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

DATASET_ENCODING = "ISO-8859-1"

TRAIN_SIZE = 0.75



# TEXT CLENAING

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



# KERAS

EMBEDDING_DIM = 100

SEQUENCE_LENGTH = None

EPOCHS = 8

BATCH_SIZE = 1024
dataset_path = '/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv'

print("Open file:", dataset_path)

df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
df.head()
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

def decode_sentiment(label):

    return decode_map[int(label)]







stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")



def preprocess(text, stem=False):

    # Remove link,user and special characters

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    return " ".join(tokens)



import sys

import regex as re



FLAGS = re.MULTILINE | re.DOTALL



def hashtag(text):

    text = text.group()

    hashtag_body = text[1:]

    if hashtag_body.isupper():

        result = " {} ".format(hashtag_body.lower())

    else:

        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])+", hashtag_body, flags=FLAGS))

    return result



def allcaps(text):

    text = text.group()

    return text.lower() + " <allcaps>"





def tokenize(text):

    # Remove stopwords.

    

    tokens = []

    for token in text.split():

        if token not in stop_words:

            tokens.append(stemmer.stem(token))

    text = " ".join(tokens)

    

    # Different regex parts for smiley faces

    eyes = r"[8:=;]"

    nose = r"['`\-]?"



    # function so code less repetitive

    def re_sub(pattern, repl):

        return re.sub(pattern, repl, text, flags=FLAGS)



    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")

    text = re_sub(r"@\w+", "<user>")

    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")

    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")

    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")

    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")

    text = re_sub(r"/"," / ")

    text = re_sub(r"<3","<heart>")

    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")

    text = re_sub(r"#\S+", hashtag)

    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")

    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")



    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.

    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)

    text = re_sub(r"([A-Z]){2,}", allcaps)



    return text.lower()

%%time

df.target = df.target.apply(lambda x: decode_sentiment(x))
%%time

df.text = df.text.apply(lambda x: tokenize(x))
df.loc[:, 'text_length'] = df.text.apply(lambda x: len(x.strip().split()))
df.head()
max_seq_len = df.text_length.max()

SEQUENCE_LENGTH = max_seq_len
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

print("TRAIN size:", len(df_train))

print("TEST size:", len(df_test))
%%time

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_train.text)



vocab_size = len(tokenizer.word_index) + 1

print("Total words", vocab_size)
%%time

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=max_seq_len)

x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=max_seq_len)
encoder = LabelEncoder()

encoder.fit(df_train.target.tolist())



y_train = encoder.transform(df_train.target.tolist())

y_test = encoder.transform(df_test.target.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("y_train",y_train.shape)

print("y_test",y_test.shape)
print("x_train", x_train.shape)

print("y_train", y_train.shape)

print()

print("x_test", x_test.shape)

print("y_test", y_test.shape)
y_train[:10]
embeddings_index = {}

with open('./glove.twitter.27B.100d.txt') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vector = np.asarray(values[1:],dtype='float32')

        embeddings_index[word] = vector

print(len(embeddings_index))
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))

for word, i in tokenizer.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

embedding_matrix.shape
from keras.layers import Input

from keras.models import Model
input_tensor = Input(shape=(64,))

x = Embedding(vocab_size, EMBEDDING_DIM, weights = [embedding_matrix], trainable = False)(input_tensor)

x = Dropout(0.5)(x)

x = Bidirectional(LSTM(32, return_sequences = True, dropout = 0.2, recurrent_dropout = 0.2))(x)

x = Bidirectional(LSTM(32, dropout = 0.4, recurrent_dropout = 0.4))(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(input_tensor, x)

model.summary()
model.compile(loss='binary_crossentropy',

              optimizer="adadelta",

              metrics=['accuracy'])
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
# %%time

# history = model.fit(x_train, y_train,

#                     batch_size=BATCH_SIZE,

#                     epochs=EPOCHS,

#                     validation_split=0.25,

#                     verbose=1,

#                     callbacks=callbacks)
# %%time

# score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

# print()

# print("ACCURACY:",score[1])

# print("LOSS:",score[0])