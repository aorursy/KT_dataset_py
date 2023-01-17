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
import fastai

from fastai import *

from fastai.text import * 

from functools import partial
# load the dataset

data = open('/kaggle/input/corpus').read()

labels, texts = [], []

for i, line in enumerate(data.split("\n")):

    content = line.split()

    labels.append(content[0])

    texts.append(" ".join(content[1:]))



# create a dataframe using texts and lables

trainDF = pd.DataFrame()

trainDF['label'] = labels

trainDF['text'] = texts

trainDF.head()
trainDF.shape
from sklearn.model_selection import train_test_split



# split data into training and validation set

df_trn, df_val = train_test_split(trainDF, stratify = trainDF['label'], test_size = 0.1, random_state = 12)
# Language model data

data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")



# Classifier model data

data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.7)
learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.save_encoder('ft_enc1')
learn = text_classifier_learner(data_clas,AWD_LSTM,drop_mult=0.7)

learn.load_encoder('ft_enc1')
data_clas.show_batch()
learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
learn.unfreeze()

learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
learn = text_classifier_learner(data_clas,AWD_LSTM,drop_mult=0.5)

learn.load_encoder('ft_enc1')
learn.unfreeze()

learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))