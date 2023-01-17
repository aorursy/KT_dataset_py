# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Dropout, LSTM, Dense, Flatten

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.embeddings import Embedding

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load train and test data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
# First 5 data

train_df.head()
# First 5 data

test_df.head()
# Features and target

X = train_df.drop("target", axis = 1)

y = train_df["target"]
# Split data into train and test 

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 34)

x_train.shape, y_train.shape, x_val.shape, y_val.shape
# Fill nan/na value with 0

x_train = x_train["question_text"].fillna("_NA_").values

x_val = x_val["question_text"].fillna("_NA_").values

x_test = test_df["question_text"].fillna("_NA_").values
# Attribute that we will use in function

max_words = 100000

max_len = 200

# it takes lot of time that's why only for 2 epoch

epoch = 1

embedding_vecor_length = 32

batch_size = 1024

# Tokenize text and select top 100k features

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)

x_val = tokenizer.texts_to_sequences(x_val)

x_test = tokenizer.texts_to_sequences(x_test)
# Padding 

x_trn_seq = pad_sequences(x_train, maxlen = max_len)

x_val_seq = pad_sequences(x_val, maxlen = max_len)

x_test = pad_sequences(x_test, maxlen = max_len)
# Build sequential model with 2 LSTM layer follwed by dropout and flatten

model = Sequential()

model.add(Embedding(max_words, embedding_vecor_length, input_length = max_len))

model.add(LSTM(100, return_sequences = True))

model.add(Dropout(0.5))



model.add(LSTM(100, return_sequences = True))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1, activation = "sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

model.summary()
# Train model that gives you history about train loss and validation loss

history = model.fit(x_trn_seq, y_train, batch_size, epochs = epoch, validation_data = (x_val_seq, y_val))

# Accuracy on validation data

score = model.evaluate(x_val_seq, y_val, verbose = 0)

print("Validation Score:", score[0])

print("Validation Accuracy", score[1])
# Due to time constraints, commented it

'''

# Plot validataion and train loss

x = range(1, epoch + 1)

val_loss = history.history["val_loss"]

train_loss = history.history["loss"]

plt.plot(x, val_loss, "b", label = "Validation loss")

plt.plot(x, train_loss, "r", label = "Train loss")

plt.xlabel("Epoch")

plt.ylabel("Categorical Crossentropy loss")

plt.legend()

plt.show()

'''
# predict results

pred = model.predict(x_test)

sample_submission = pd.DataFrame({"qid": test_df["qid"].values})

sample_submission["prediction"] = pred

sample_submission.to_csv("submission.csv", index = False)