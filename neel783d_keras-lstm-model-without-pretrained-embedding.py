# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing import text, sequence

from keras import backend as K

from keras.models import load_model

import keras

import pickle

print(K.tensorflow_backend._get_available_gpus())

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train_data = train["comment_text"]

label_data = train["target"]

test_data = test["comment_text"]

train_data.shape, label_data.shape, test_data.shape
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train_data) + list(test_data))
train_data = tokenizer.texts_to_sequences(train_data)

test_data = tokenizer.texts_to_sequences(test_data)
MAX_LEN = 200

train_data = sequence.pad_sequences(train_data, maxlen=MAX_LEN)

test_data = sequence.pad_sequences(test_data, maxlen=MAX_LEN)
max_features = None
max_features = max_features or len(tokenizer.word_index) + 1

max_features
type(train_data), type(label_data.values), type(test_data)

label_data = label_data.values
# Keras Model

# Model Parameters

NUM_HIDDEN = 256

EMB_SIZE = 256

LABEL_SIZE = 1

MAX_FEATURES = max_features

DROP_OUT_RATE = 0.2

DENSE_ACTIVATION = "sigmoid"

NUM_EPOCH = 1



# Optimization Parameters

BATCH_SIZE = 1000

LOSS_FUNC = "binary_crossentropy"

OPTIMIZER_FUNC = "adam"

METRICS = ["accuracy"]



class LSTMModel:

    

    def __init__(self):

        self.model = self.build_graph()

        self.compile_model()

    

    def build_graph(self):

        model = keras.models.Sequential([

            keras.layers.Embedding(MAX_FEATURES, EMB_SIZE),

            keras.layers.CuDNNLSTM(NUM_HIDDEN),

            keras.layers.Dropout(rate=DROP_OUT_RATE),

            keras.layers.Dense(LABEL_SIZE, activation=DENSE_ACTIVATION)])

        return model

    

    def compile_model(self):

        self.model.compile(

            loss=LOSS_FUNC,

            optimizer=OPTIMIZER_FUNC,

            metrics=METRICS)
model = LSTMModel().model

model.fit(

    train_data, 

    label_data, 

    batch_size = BATCH_SIZE, 

    epochs = NUM_EPOCH)
submission_in = '../input/sample_submission.csv'

submission_out = 'submission.csv'
result = model.predict(test_data)
submission = pd.read_csv(submission_in, index_col='id')

submission['prediction'] = result

submission.reset_index(drop=False, inplace=True)
submission.head()
submission.to_csv(submission_out, index=False)