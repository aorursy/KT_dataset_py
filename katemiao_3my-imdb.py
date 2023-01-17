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
# Set your own project id here

# PROJECT_ID = 'your-google-cloud-project'

  

# from google.cloud import bigquery

# client = bigquery.Client(project=PROJECT_ID, location="US")



%matplotlib inline

from fastai.text import * 
print(data.vocab.itos[:12])
# but the underlying data is all numbers

data.train_ds[2][0].data[:10]
# with the data block API will be more flexible
path = untar_data(URLs.IMDB)

path.ls()
! cat '/tmp/.fastai/data/imdb/README'
(path/'train').ls()
(path/'train'/'pos').ls()[:5]
! cat '/tmp/.fastai/data/imdb/train/pos/10731_7.txt'
path.ls()
# language model can use a lot of GPU, may need to decrease batchsize here

# bs = 64

bs = 32
data_lm = (TextList.from_folder(path)

          .filter_by_folder(include=['train','test','unsup'])

          .split_by_rand_pct(0.1)

          .label_for_lm()

          .databunch(bs=bs))
data_lm.save('data_lm.pkl')
data_lm = load_data(path, 'data_lm.pkl', bs=bs)
data_lm.show_batch()
print(data_lm.vocab.itos[:12])
data_lm.train_ds[1][0]
data_lm.train_ds[1][0].data[:10]
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
learn.save('/kaggle/working/fit_head')
learn.save('fit_head')
TEXT = 'I liked this moive because'

N_WORDS = 40
# result before fine-tune

for _ in range(2):

    print(learn.predict(TEXT, N_WORDS, temperature=0.75))

    print('\n')
learn.unfreeze()
learn.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))
learn.save_encoder('/kaggle/working/fine_tuned_enc')
learn.save_encoder('fine_tuned_enc')
path = untar_data(URLs.IMDB)
path.ls()
data_lm.vocab.itos[:4]
data_class = (TextList.from_folder(path, vocab=data_lm.vocab)

             .split_by_folder(valid='test')

             .label_from_folder(classes=['neg','pos'])

             .databunch(bs=bs))
data_class.save('data_class.pkl')
data_class = load_data(path, 'data_class.pkl', bs=bs)
data_class.show_batch()
learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.predict("I really loved that movie, it was awesome!")