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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
qid_test = pd.read_csv('../input/test.csv')[['qid']]
sample_submission = pd.read_csv('../input/sample_submission.csv')
import tensorflow as tf

from sklearn.model_selection import train_test_split
import fastai

from fastai import *

from fastai.text import * 

import pandas as pd

import numpy as np

from functools import partial

import io

import os
from sklearn.metrics import f1_score
df = pd.DataFrame({'label':train.target, 'text':train.question_text})
test = pd.DataFrame({'text':test.question_text})
# split data into training and validation set

df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.25, random_state = 42)
test.head()
df_trn.head()
# Language model data

data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, test_df=test, path = "", text_cols='text', label_cols='label')



# Classifier model data

data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, test_df = test,text_cols='text',label_cols='label', vocab=data_lm.train_ds.vocab, bs=32)
learn = language_model_learner(data_lm, AWD_LSTM)
moms = (0.8,0.7)
learn.unfreeze()

learn.fit_one_cycle(3, slice(1e-2), moms=moms)
learn.save_encoder('enc')
learn = text_classifier_learner(data_clas, AWD_LSTM)

learn.load_encoder('enc')

learn.fit_one_cycle(1, moms=moms)
res = learn.get_preds(ds_type=DatasetType.Valid)
# check f1 score on valid

predictions = np.argmax(res[0], axis=1)

f1_score(res[1], predictions) # 0.537
pd.crosstab(predictions, res[1])
# check on test set

test_res = learn.get_preds(ds_type=DatasetType.Test)
test_pred = np.argmax(test_res[0], axis=1)
submission_df = pd.DataFrame({'qid':qid_test.qid, 'prediction':test_pred})
submission_df.to_csv('submission.csv', index=False)