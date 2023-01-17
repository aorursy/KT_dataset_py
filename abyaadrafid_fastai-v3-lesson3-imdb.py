# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.text import *
bs=48
path=untar_data(URLs.IMDB)
data_lm=(TextList.from_folder(path)

        .filter_by_folder(include=['train', 'test', 'unsup']) 

        .split_by_rand_pct(0.1)

        .label_for_lm()  

        .databunch(bs=bs))

data_lm.save('data_lm.pkl')
data_lm = load_data(path, 'data_lm.pkl', bs=bs)
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot()
learn.save('fit_head')
learn.fit_one_cycle(2, 1e-2, moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "I liked this movie because"

N_WORDS = 40

N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)

             #grab all the text files in path

             .split_by_folder(valid='test')

             #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)

             .label_from_folder(classes=['neg', 'pos'])

             #label them all with their folders

             .databunch(bs=bs))



data_clas.save('data_clas.pkl')
data_clas = load_data(path, 'data_clas.pkl', bs=bs)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
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
learn.predict("I really loved that movie, it was awesome!")