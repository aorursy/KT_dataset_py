!pip3 install fastai==1.0.42
from fastai.text import * 

from fastai.gen_doc.nbdoc import *

from fastai.datasets import * 

from fastai.datasets import Config

from pathlib import Path

import pandas as pd
path = untar_data(URLs.IMDB_SAMPLE)

path
df = pd.read_csv(path/'texts.csv')

df.head()
# Language model data

data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')

# Classifier model data

data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)

learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.predict("This is a review about", n_words=10)
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, drop_mult=0.5)

learn.load_encoder('ft_enc')
data_clas.show_batch()
learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)

learn.fit_one_cycle(5, slice(5e-3/2., 5e-3))

learn.unfreeze()

learn.fit_one_cycle(5, slice(2e-3/100, 2e-3))
learn.predict("This was a great movie!")