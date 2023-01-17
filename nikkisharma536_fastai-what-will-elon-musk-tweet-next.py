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

data = pd.read_csv("../input/data_elonmusk.csv",  encoding='latin1')

data.head()
data = (TextList.from_df(data, cols='Tweet')

                .random_split_by_pct(0.1)

               .label_for_lm()  

                .databunch(bs=48))
data.show_batch()
learn = language_model_learner(data, AWD_LSTM, drop_mult=0.3, model_dir = '/tmp/work')
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(10, 1e-2)
# Lets tune a little more

learn.unfreeze()

learn.fit_one_cycle(1, 5e-3, moms=(0.8,0.7))

# Lets tune a little more

learn.unfreeze()

learn.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))
learn.model
N_WORDS = 20

N_SENTENCES = 5
learn.predict("Clean energy will be", N_WORDS, temperature=0.75)
print("\n".join(learn.predict("Clean energy will be", N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
print("\n".join(learn.predict("Climate change will", N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
print("\n".join(learn.predict("Tesla is the", N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))