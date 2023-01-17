# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import codecs

from tqdm import tqdm



from nltk.tokenize import RegexpTokenizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, GlobalMaxPool1D, Bidirectional, GlobalMaxPooling1D

from keras.layers import LSTM, GRU, Dropout , BatchNormalization, Embedding, Flatten, GlobalAveragePooling1D, concatenate, Input

!pip install 'keras_tqdm'

from keras_tqdm import TQDMNotebookCallback

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Conv1D,Flatten,Concatenate

from keras.models import Model

from keras import optimizers

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score

df = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')
import nltk

nltk.download('stopwords')
sentences_train = df["comment_text"].fillna("_na_").values

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = df[classes].values

sentences_test = test["content"].fillna("_na_").values
# Embedding parameter set

embed_size = 100 # how big is each word vector

max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 50 # max number of words in a comment to use
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(sentences_train))

tokens_train = tokenizer.texts_to_sequences(sentences_train)

tokens_test = tokenizer.texts_to_sequences(sentences_test)

X_train = pad_sequences(tokens_train, maxlen=maxlen)

X_test = pad_sequences(tokens_test, maxlen=maxlen)
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = LSTM(4, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(x)

x = Conv1D(16,4,activation='relu')(x)

x = Flatten()(x)

x = Dense(100, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr = 0.001,decay = 1e-06), metrics=['accuracy'])

filepath="toxic_comment/weights-improvement.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

model.summary()
model.fit(X_train, y, batch_size=32, epochs=5,callbacks=callbacks_list, verbose=1, validation_split=0.2)
y_test=model.predict(X_test)
sample_submission=pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

sample_submission[classes]=y_test

sample_submission.to_csv("submission.csv",index=False)
from keras.models import model_from_json

import os

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
# # load json and create model

# json_file = open('model.json', 'r')

# loaded_model_json = json_file.read()

# json_file.close()

# loaded_model = model_from_json(loaded_model_json)

# # load weights into new model

# loaded_model.load_weights("model.h5")

# print("Loaded model from disk")