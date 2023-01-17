# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# Inspriration:https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-069 
# Importing necessary packages:
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Conv1D,Flatten,Concatenate
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
# Mentioning the name of the input file
TRAIN_DATA_FILE= '../input/train.csv'
TEST_DATA_FILE= '../input/test.csv'
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
# Little preprocessing required
sentences_train = train["comment_text"].fillna("_na_").values
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[classes].values
sentences_test = test["comment_text"].fillna("_na_").values
# Embedding parameter set
embed_size = 100 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a comment to use
#similar use : https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-069
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
model.summary()
model.fit(X_train, y, batch_size=32, epochs=3,verbose=1, validation_split=0.2)
y_test = model.predict(X_test)
Submit = pd.DataFrame(test.id,columns=['id'])
Submit2 = pd.DataFrame(y_test,columns=classes)
Submit = pd.concat([Submit,Submit2],axis=1)
Submit.to_csv("Kaggle_Submission_Convolution_LSTM_.csv",index=False)
