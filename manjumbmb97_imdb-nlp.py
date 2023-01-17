import torch

import torch.nn as nn

import fastai

from fastai.text import *

import pandas as pd



import os

print(os.listdir('../input/'))
path = untar_data(URLs.IMDB_SAMPLE)

path
df = pd.read_csv(path/'texts.csv')

df.head()
fig = plt.figure()

df = df.drop('is_valid', axis=1)

df.groupby('label').count().plot.bar(ylim=0)

plt.show()
# Language model data

data_lm = TextLMDataBunch.from_csv(path, 'texts.csv', device='cuda')

# Classifier model data

data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, device='cuda', bs=32)
print(data_lm)
data_lm.save('../data_lm_export.pkl')

data_clas.save('../data_clas_export.pkl')
data_lm = load_data(path, '../data_lm_export.pkl')

data_clas = load_data(path, '../data_clas_export.pkl', bs=16)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, model_dir="../")

learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit(4, 1e-3)
learn.predict("This is a review about", n_words=10)
learn.save_encoder('../ft_enc')
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3, model_dir="../")

learn.load_encoder('../ft_enc')
data_clas.show_batch()
learn.fit_one_cycle(5, 1e-3)
preds,y,losses = learn.get_preds(with_loss=True)
learn.predict("Kind of drawn in by the erotic scenes, only to realize this was one of the most amateurish and unbelievable bits of film I've ever seen. Sort of like a high school film project. What was Rosanna ")
learn.predict("What an absolutely stunning movie, if you have 2.5 hrs to kill, watch it, you won't regret it, it's too much fun! Rajnikanth carries the movie on his shoulders")
interp = ClassificationInterpretation(learn, preds, y, losses)

interp.plot_confusion_matrix()