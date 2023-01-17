import numpy as np

import pandas as pd

import regex as re

import os

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from fastai.text import *



import re



# view the contents of the input directory

print(os.listdir("../input"))
bbc_data = pd.read_csv('../input/bbc-text.csv')

print('The dataset has {} observations'.format(bbc_data.shape[0]))

bbc_data.head(10)
bbc_data['category'].value_counts()
np.random.seed(1001)

test_idx = np.random.choice(range(bbc_data.shape[0]), size = 250)





train_df = bbc_data.iloc[[i for i in bbc_data.index if i not in test_idx]]

test_df = bbc_data.iloc[test_idx]



train_df.shape, test_df.shape
# create the data bunch for fine-tuning the language model

lm_data = TextLMDataBunch.from_df('../working', train_df = train_df, valid_df = test_df, text_cols='text')
# checking the data

lm_data.show_batch()
# actually tune the language model

learn = language_model_learner(lm_data, AWD_LSTM, drop_mult = 0.5)

learn.fit_one_cycle(1)
# fit a couple more with a lower learning rate

learn.fit_one_cycle(8, 1e-3)
# see how it performs - it's a bit silly, but it should be better than not tuning

learn.predict("Just this evening, our reporters were informed", n_words=10)
# save the language model

learn.save_encoder('bbc_lm')
clas_data = TextClasDataBunch.from_df('../working', train_df = train_df, valid_df = test_df, text_cols='text', label_cols='category')
clas_data.show_batch()
learn = text_classifier_learner(clas_data, AWD_LSTM, drop_mult = 0.75)

learn.load_encoder('bbc_lm')
# find a good learning rate

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(15, 3e-2)
# needs a few more now that we're using more dropout

learn.fit_one_cycle(7, 1e-2)
# i'm cutting it off here, but it's clearly still learning

learn.fit_one_cycle(10, 1e-2)
exp_i = 32



print('''Example:

- Predicted: {}

- Actual: {}

- Text: {}

'''.format(learn.predict(test_df['text'].iloc[exp_i])[0],

           test_df['category'].iloc[exp_i],

           test_df['text'].iloc[exp_i]))