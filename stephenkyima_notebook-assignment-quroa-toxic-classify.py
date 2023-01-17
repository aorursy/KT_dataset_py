# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from sklearn.model_selection import train_test_split



import warnings

warnings.simplefilter("ignore")



train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

train_df.info()
# 保留65万的数据作为训练集

df_train = train_df.loc[:650000, :]

df_train, df_valid = train_test_split(df_train, test_size=0.1)



print(df_train.target.values[:10])

# print(df_train.head(n=10))

print(df_train.head(n=3))
# backend

print(df_train.columns)



# create a Vocabulary using the question_text

from keras.preprocessing.text import Tokenizer

def get_vocab(df, num_words=20000):

    """

    get the dictionary using the df

    """

    tokenizer = Tokenizer(num_words=num_words)

    texts = df.question_text.tolist()

    tokenizer.fit_on_texts([item.lower() for item in texts])

    return tokenizer

# Tokenzier by using dictionary on training data

tokenizer = get_vocab(df_train)
%%time

from keras.preprocessing.sequence import pad_sequences



MAX_LENGTH = 40

train_X = np.array(pad_sequences(tokenizer.texts_to_sequences(df_train.question_text.tolist()), maxlen=MAX_LENGTH, padding = 'post'))

valid_X = np.array(pad_sequences(tokenizer.texts_to_sequences(df_valid.question_text.tolist()), maxlen=MAX_LENGTH, padding = 'post'))

print(train_X.shape)



train_y, valid_y = np.array(df_train.target.values), np.array(df_valid.target.values)
# using embedding here to get the numpy array for later useage

embeddings_index = {}

file = open('../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')

for line in tqdm(file):

    values = line.split(" ")

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

file.close()



print('Found %s word vectors.' % len(embeddings_index))
vocab_size = len(tokenizer.word_index.items())

# Create a weight matrix for words in training docs

embedding_matrix = np.random.normal(loc=0, scale=1.0, size=(vocab_size+1, 300))

for word, i in tokenizer.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# Building a model with using Keras

from keras.models import Sequential

from keras.layers import RNN, LSTM, Dropout, Flatten, Embedding, SpatialDropout1D, Dense



# define model

model = Sequential()

model.add(Embedding(vocab_size+1, 300, input_length=MAX_LENGTH, weights=[embedding_matrix]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(train_X, train_y, epochs=2, verbose=1, batch_size=256)
# Prediction

df_test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

test_tokenizer = get_vocab(df_test)

test_X = np.array(pad_sequences(tokenizer.texts_to_sequences(df_test.question_text.tolist()), maxlen=MAX_LENGTH, padding = 'post'))



pred_y = model.predict([test_X], batch_size=256, verbose=1)
pred_y = (pred_y > 0.35).astype(int)

submit_pd = pd.DataFrame({"qid":df_test["qid"].values})

submit_pd['prediction'] = pred_y

submit_pd.to_csv("submission.csv", index=False)