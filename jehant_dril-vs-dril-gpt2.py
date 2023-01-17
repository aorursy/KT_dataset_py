%reload_ext autoreload

%autoreload 2

%matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.text import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



filename = "/kaggle/input/dril-tweets/dril_tweets.csv"

dril = pd.read_csv(filename)



filename = "/kaggle/input/dril-gpt2-tweets/dril_gpt2_tweets.csv"

dril_gpt2 = pd.read_csv(filename)



dril.head(), dril_gpt2.head()
# Optionally balance number from each



# n = dril_gpt2.shape[0]

# dril = dril.sample(n=n, random_state=42)

# dril.shape == dril_gpt2.shape
dril.insert(0, 'label', 'dril')

dril
dril_gpt2.insert(0, 'label', 'dril_gpt2')

dril_gpt2
raw_data = dril.append(dril_gpt2, ignore_index=True).sample(frac=1, random_state=42)

raw_data
train_df = raw_data.sample(frac=0.8, random_state=42)

valid_df = raw_data.drop(train_df.index)



data = TextLMDataBunch.from_df('.', train_df=train_df, valid_df=valid_df, num_workers=0)

data.save()
data.show_batch(rows=100)
data.vocab.itos[:100]
data.train_ds[80][0]
learn = language_model_learner(data, AWD_LSTM, drop_mult=0.3)

# URLs
learn.lr_find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(4, 2e-2, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "FUCK"

N_WORDS = 40

N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
data_clas = TextDataBunch.from_df('.', train_df=train_df, valid_df=valid_df, num_workers=0)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 3e-2, moms=(0.8,0.7))
learn.save('first')
learn.load('first');
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.save('second')
learn.load('second');
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.save('third')
learn.load('third');
learn.unfreeze()

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))