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
data = pd.read_csv('/kaggle/input/titlecmpnothers/unique_CJO_3class.csv')

data.head()
data.info()
data.dropna(inplace = True)
data.info()
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(x='label',data=data)
fast_data = data[['text','label']]
from pathlib import Path

from fastai.text import *
fast_data = fast_data.sample(frac=1).reset_index(drop=True)

fast_data.shape[0]*.9
train_df, valid_df = fast_data.loc[:800000,:],fast_data.loc[800000:,:]
train_df.head()
path =Path(".")
data_lm = TextLMDataBunch.from_df(path, train_df, valid_df, text_cols=['text'], bs=64)

data_clas = TextClasDataBunch.from_df(path, train_df, valid_df, text_cols=['text'], label_cols=['label'], bs=64)
data_lm.show_batch()
data_clas.show_batch()
learn = language_model_learner(data_lm, arch = AWD_LSTM, pretrained = True, drop_mult=0.2)

learn.lr_find() # find learning rate

learn.recorder.plot() # plot learning rate graph
learn.fit_one_cycle(3, 1e-2)
learn.unfreeze() # must be done before calling lr_find

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, 1e-3)
learn.save_encoder('word-enc')
learn = text_classifier_learner(data_clas, arch = AWD_LSTM, pretrained = True, drop_mult=0.2)

learn.load_encoder('word-enc')



# find and plot learning rate

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2)



# unfreeze one layer group and train another epoch

learn.freeze_to(-3)

learn.fit_one_cycle(3, slice(5e-3/2., 5e-3))



learn.unfreeze()

learn.fit_one_cycle(3, slice(2e-3/100, 2e-3))
learn.export(file = Path("/kaggle/working/export.pkl"))



print(learn.predict("Software Engineer")[0])
print(learn.predict("Lineupx")[0])