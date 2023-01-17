# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from functools import partial

import time
start_time = time.time()

from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

import logging
from keras.callbacks import Callback

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from eli5.lime import TextExplainer
# Any results you write to the current directory are saved as output.
tweets = pd.read_csv("../input/clinton-trump-tweets/tweets.csv")
tweets.head()
tweets.shape
sum(tweets.text.isnull())
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
raw_text = tweets["text"].str.lower()
max_features = 130000
max_len = 220
tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text)
tweets["comment_seq"] = tk.texts_to_sequences(raw_text)
tweets_pad_sequences = pad_sequences(tweets.comment_seq, maxlen = max_len)
tweets_pad_sequences.shape
model = load_model("../input/bi-gru-lstm-cnn-poolings-fasttext/best_model.hdf5")
pred = model.predict(tweets_pad_sequences, batch_size = 1024, verbose = 1)
pred.max()
toxic_predictions = pd.DataFrame(columns=list_classes, data=pred)
toxic_predictions.head()
toxic_predictions['id'] = tweets['id'].values
toxic_predictions['handle'] = tweets['handle'].values
toxic_predictions['text'] = tweets['text'].values
toxic_predictions.tail()
Hillary_predictions = toxic_predictions[toxic_predictions['handle'] == 'HillaryClinton']
Trump_predictions = toxic_predictions[toxic_predictions['handle'] == 'realDonaldTrump']
Hillary_predictions[list_classes].describe()
Trump_predictions[list_classes].describe()
def predict_texts(texts, class_idx):
    sequence = tk.texts_to_sequences(texts)
    sequence = pad_sequences(sequence, maxlen=max_len) 
    preds = model.predict(sequence, batch_size=100, verbose=1)[:, class_idx]
    # Make the probability sums to 1
    preds = np.array([preds, 1-preds]).transpose()
    return preds

def explain_text(text, class_idx):
    te = TextExplainer(random_state=42, n_samples=1000)
    te.fit(text, partial(predict_texts, class_idx=class_idx))
    print(te.metrics_)
    return te.show_prediction(target_names=[list_classes[class_idx], "None"])
class_idx = 0
print(list_classes[class_idx])
explain_text(Hillary_predictions.loc[Hillary_predictions['toxic'].idxmax()]['text'], class_idx=class_idx)
Hillary_predictions.loc[Hillary_predictions['toxic'].idxmax()]
class_idx = 3
print(list_classes[class_idx])
explain_text(Hillary_predictions.loc[Hillary_predictions['threat'].idxmax()]['text'], class_idx=class_idx)
Hillary_predictions.loc[Hillary_predictions['threat'].idxmax()]
class_idx = 0
print(list_classes[class_idx])
explain_text(Hillary_predictions.loc[Hillary_predictions['threat'].idxmax()]['text'], class_idx=class_idx)
Hillary_predictions.loc[Hillary_predictions['insult'].idxmax()]
class_idx = 4
print(list_classes[class_idx])
explain_text(Hillary_predictions.loc[Hillary_predictions['insult'].idxmax()]['text'], class_idx=class_idx)
Hillary_predictions.loc[Hillary_predictions['identity_hate'].idxmax()]
class_idx = 5
print(list_classes[class_idx])
explain_text(Hillary_predictions.loc[Hillary_predictions['identity_hate'].idxmax()]['text'], class_idx=class_idx)
Trump_predictions.loc[Trump_predictions['toxic'].idxmax()]
class_idx = 0
print(list_classes[class_idx])
explain_text(Trump_predictions.loc[Trump_predictions['toxic'].idxmax()]['text'], class_idx=class_idx)
Trump_predictions.loc[Trump_predictions['threat'].idxmax()]
class_idx = 3
print(list_classes[class_idx])
explain_text(Trump_predictions.loc[Trump_predictions['threat'].idxmax()]['text'], class_idx=class_idx)
Trump_predictions.loc[Trump_predictions['insult'].idxmax()]
class_idx = 4
print(list_classes[class_idx])
explain_text(Trump_predictions.loc[Trump_predictions['insult'].idxmax()]['text'], class_idx=class_idx)
Trump_predictions.loc[Trump_predictions['identity_hate'].idxmax()]
class_idx = 5
print(list_classes[class_idx])
explain_text(Trump_predictions.loc[Trump_predictions['identity_hate'].idxmax()]['text'], class_idx=class_idx)
