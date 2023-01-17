# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import os



# Time

import time

import datetime



# Numerical

import numpy as np

import pandas as pd



# Tools

import itertools

from collections import Counter



# NLP

import re

import nltk

from nltk.corpus import stopwords



# Preprocessing

from sklearn import preprocessing

from sklearn.utils import class_weight as cw

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



# Model Selection

from sklearn.model_selection import train_test_split



# Evaluation Metrics

from sklearn import metrics 

from sklearn.metrics import f1_score, accuracy_score,confusion_matrix,classification_report



# Deep Learing Preprocessing - Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical



# Deep Learning Model - Keras

from keras.models import Model

from keras.models import Sequential



from keras.layers import Dense, Embedding

from keras.models import Sequential



# Deep Learning Model - Keras - RNN

from keras.layers import Embedding, LSTM, Bidirectional



# Deep Learning Model - Keras - General

from keras.layers import Input, Add, concatenate, Dense, Activation, BatchNormalization, Dropout, Flatten

from keras.layers import LeakyReLU, PReLU, Lambda, Multiply



from keras.preprocessing import sequence

from keras import regularizers



# Deep Learning Parameters - Keras

from keras.optimizers import RMSprop, Adam



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



from fastai.imports import *

from fastai.text import *
path = Path(os.path.abspath(os.curdir))
#File Import

filepath = Path('../input')

df = pd.read_csv(filepath/'Tweets.csv')

df.head()
df = df[['airline_sentiment','text']]

df.head()
train = df[:int(len(df)*.99)]

valid = df[int(len(df)*.99):]
lm_dat = TextLMDataBunch.from_df(path, train, valid)

lm_dat.save('data_lm_export.pkl')
lm_learn = language_model_learner(lm_dat, AWD_LSTM, drop_mult=0.4)
lm_learn.lr_find()

lm_learn.recorder.plot()
lm_learn.fit_one_cycle(4, 1e-2)
lm_learn.unfreeze()

lm_learn.lr_find(); lm_learn.recorder.plot()
lm_learn.fit_one_cycle(4, 1e-3)

#Encoder

lm_learn.save_encoder('ft_enc')
#Splitting the dataset in 80:20 ratio

train = df[:int(len(df)*.80)]

valid = df[int(len(df)*.80):]
# Classifier model data

data_clas = TextClasDataBunch.from_df(path, train, valid, vocab=lm_dat.train_ds.vocab, bs=32)

data_clas.save('data_clas_export.pkl') ; data_clas = load_data(path, 'data_clas_export.pkl', bs=16)
#Building a classifier with fine-tuned encoder 

lm_learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=.3, metrics=[accuracy,Precision(average='weighted'),Recall(average='weighted')])

lm_learn.load_encoder('ft_enc')
data_clas.show_batch()
lm_learn.lr_find()

lm_learn.recorder.plot()
lm_learn.fit_one_cycle(4, 1e-2)
lm_learn.freeze_to(-2)

lm_learn.fit_one_cycle(4, slice(1e-3/(2.6**4), 1e-3))
#unfreezing the model and fine-tuning it

lm_learn.unfreeze()

lm_learn.fit_one_cycle(8, slice(1e-5/(2.6**4),1e-5))

lm_learn.save('final')
#Obtaining Test Accuracy

valid['pred_sentiment'] = valid['text'].apply(lambda row: str(lm_learn.predict(row)[0]))

print("Test Accuracy: ", accuracy_score(valid['airline_sentiment'], valid['pred_sentiment']))