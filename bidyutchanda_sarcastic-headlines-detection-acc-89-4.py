# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/fastai-wt103/wt103/wt103"))



# Any results you write to the current directory are saved as output.
data_df = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines = True)

data_df.head()
from fastai import *

from fastai.text import *

from sklearn.model_selection import train_test_split
#Splitting into training and validation sets

df_train, df_valid = train_test_split(data_df, stratify=data_df['is_sarcastic'], test_size = 0.25, random_state=42)

#For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, 

#stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
# Language model data

data_lm = TextLMDataBunch.from_df(train_df = df_train, valid_df = df_valid, path = "", text_cols = 'headline', label_cols = 'is_sarcastic')

                                 
# Classifier model data

data_clas = TextClasDataBunch.from_df(train_df = df_train, valid_df = df_valid, path = "", text_cols = 'headline', label_cols = 'is_sarcastic', vocab = data_lm.train_ds.vocab, bs=32)
learn = language_model_learner(data_lm, AWD_LSTM)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

moms = (0.8,0.7)
learn.fit_one_cycle(4, slice(1e-02,1e-01), moms=moms)
learn.save_encoder('enc')
learn = text_classifier_learner(data_clas, AWD_LSTM)

learn.load_encoder('enc')

learn.fit_one_cycle(1,moms=moms)
learn.save('first')
learn.load('first')

learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=moms)
learn.save('second')
learn.load('second')

learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=moms)
learn.save('third')

learn.load('third')

learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-2/(2.6**4),1e-2), moms=moms)
learn.save('fourth')

pred_class = learn.predict('Paul Pogba is the star at the moment')
pred_class
pred_class = learn.predict('Chowkidaar hi chor hai')
pred_class
data_clas.classes