!pip3 install fastai==1.0.42
#import dataset

from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

documents = dataset.data
# import libraries

import fastai

from fastai import *

from fastai.text import * 

import pandas as pd

import numpy as np

from functools import partial

import io

import os

dataset.target_names
# create a dataframe

df = pd.DataFrame({'label':dataset.target,

                   'text':dataset.data})

df.head()
#df = df[df['label'].isin([1,10])]

df = df.reset_index(drop = True)
df['label'].value_counts()
df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")

import nltk

nltk.download('stopwords')
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
df.head()
from sklearn.model_selection import train_test_split



# split data into training and validation set

df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.4, random_state = 12)
df_trn.shape, df_val.shape

# Language model data

data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")



# Classifier model data

data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)
# train the learner object

learn.fit_one_cycle(1, 1e-1)

learn.lr_find()
learn.recorder.plot()
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, drop_mult=0.7)

learn.load_encoder('ft_enc')
learn.lr_find()

learn.recorder.plot()
1e-2
#0.121962
learn.fit_one_cycle(10, 1e-2)
# get predictions

preds, targets = learn.get_preds()



predictions = np.argmax(preds, axis = 1)

pd.crosstab(predictions, targets)
#interp = ClassificationInterpretation.from_learner(learn)

#interp.plot_top_losses(9, figsize=(15, 11))

#interp.plot_confusion_matrix()