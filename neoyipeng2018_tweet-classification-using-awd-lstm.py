import os

import numpy as np

import pandas as pd

from fastai.text import * 
#Setting path for learner

path = Path(os.path.abspath(os.curdir))
#reading into pandas and renaming columns for easier api access

filepath = Path('../input')

df = pd.read_csv(filepath/'Tweets.csv')

df.rename(columns={'airline_sentiment':'label'},inplace=True)



df = df[['label','text']]

df.head(2)
# 99-1 split into train/validation set.

train = df[:int(len(df)*.99)]

valid = df[int(len(df)*.99):]
# Language model data

data_lm = TextLMDataBunch.from_df(path, train, valid)

data_lm.save('data_lm_export.pkl')

data_lm = load_data(path, 'data_lm_export.pkl')
#Training a language model, i.e. to predict the next few words

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=1.5)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 1e-2)

#learn.save('fit_head'); learn.load('fit_head')
learn.unfreeze()

learn.lr_find(); learn.recorder.plot()
learn.fit_one_cycle(8, 1e-3)

learn.save_encoder('ft_enc')
learn.predict("I booked a ticket", n_words=10)
# 80-20 split into train/validation set.

train = df[:int(len(df)*.80)]

valid = df[int(len(df)*.80):]
# Classifier model data

data_clas = TextClasDataBunch.from_df(path, train, valid, vocab=data_lm.train_ds.vocab, bs=32)

data_clas.save('data_clas_export.pkl') ; data_clas = load_data(path, 'data_clas_export.pkl', bs=16)
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=.3, metrics=[accuracy,Precision(average='weighted'),Recall(average='weighted')])

learn.load_encoder('ft_enc')
data_clas.show_batch()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 1e-2)

#learn.save('clas_head'); learn.load('clas_head')
learn.freeze_to(-2)

learn.fit_one_cycle(4, slice(1e-3/(2.6**4), 1e-3))

#learn.save('second'); learn.load('second')
learn.freeze_to(-3)

learn.fit_one_cycle(4, slice(1e-4/(2.6**4), 1e-4))

#learn.save('third') ; learn.load('third')
learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-5/(2.6**4),1e-5))

learn.save('final')
learn.predict("JetBlue gave me bad service. My media console was spoilt")
learn.predict("JetBlue was okay, nothing special")
learn.predict("Missed my flight but JetBlue gave me a great hotel voucher and food claims.Thanks!")