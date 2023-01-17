# !pip install fastai

# import libraries

import fastai

from fastai import *

from fastai.text import * 

import pandas as pd

import numpy as np

from functools import partial

import io

import os
df_imdb = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df_imdb.head()
data_lm = (TextList.from_df(df_imdb)

           #Inputs: all the text files in path

            .split_by_rand_pct(0.20)

           #We randomly split and keep 20% for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=128))

data_lm.save('tmp_lm')
data_lm.show_batch()
# Language model AWD_LSTM

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
print('Model Summary:')

print(learn.layer_groups)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, 1e-2)

learn.save('lm_hyper')
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.save_encoder('ft_enc')

data_clas = (TextList.from_df(df_imdb, cols=["review"], vocab=data_lm.vocab)

             .split_by_rand_pct(0.20)

             .label_from_df('sentiment')

             .databunch(bs=128))



data_clas.save('tmp_class')

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc')
learn.freeze_to(-1)

learn.summary()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, 1e-3)
learn.save('stage1')
learn.load('stage1')

learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.save('stage2')
from fastai.vision import ClassificationInterpretation



interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
interp = TextClassificationInterpretation.from_learner(learn)

interp.show_top_losses(10)
learn.export()

learn.model_dir = "/kaggle/working"

learn.save("stage-1",return_path=True)