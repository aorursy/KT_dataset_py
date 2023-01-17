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
# import libraries

import fastai

from fastai import *

from fastai.text import * 

import pandas as pd

import numpy as np

from functools import partial

import io

import os
dataset = pd.read_csv('../input/train.csv')

dataset = dataset.dropna()

dataset = dataset.reset_index(drop=True)



Reviews = dataset['Review Text']

dataset.topic.unique()
df = pd.DataFrame({'label':dataset.topic,

                   'text':dataset['Review Text']})
df.shape


macronum=sorted(set(df.label))

macro_to_id = dict((note, number) for number, note in enumerate(macronum))



def fun(i):

    return macro_to_id[i]



df.label=df.label.apply(fun)



df.label.value_counts()

df

df = df[df['label'].isin([1,10])]

df = df.reset_index(drop = True)
df['label'].value_counts()
df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")



from nltk.corpus import stopwords 

stop_words = stopwords.words('english')



# tokenization 

tokenized_doc = df['text'].apply(lambda x: x.split())



# remove stop-words 

tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words]) 



# de-tokenization 

detokenized_doc = [] 

for i in range(len(df)): 

    t = ' '.join(tokenized_doc[i]) 

    detokenized_doc.append(t) 

df['text'] = detokenized_doc
df.text[1]
from sklearn.model_selection import train_test_split



# split data into training and validation set

df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.25, random_state = 12)
df_trn.shape, df_val.shape
# Language model data

data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")



# Classifier model data

data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)
learn = language_model_learner(data_lm,arch = AWD_LSTM, drop_mult=0.7)

# train the learner object

learn.fit_one_cycle(2, 1e-1)
learn.lr_find()
learn.recorder.plot()

learn.save_encoder('ft_enc')

learn = text_classifier_learner(data_clas,arch = AWD_LSTM, drop_mult=0.7)

learn.load_encoder('ft_enc')
learn.lr_find()

learn.recorder.plot()


learn.fit_one_cycle(2, 1e-2)