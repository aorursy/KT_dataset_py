!pip install fastai==1.0.52
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
!mkdir stuff
import fastai

from fastai import * 

from fastai.text import *

import random

import torch
'''Setting seed for reproducibility'''

SEED = 2019



# python RNG

random.seed(SEED)



# pytorch RNGs

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)



# numpy RNG

np.random.seed(SEED)
train_df = pd.read_table('../input/train.tsv')

test_df = pd.read_table('../input/test.tsv')
train_df.head()
train_df.isnull().sum()
train_df['Sentiment'].value_counts()
train_df = pd.DataFrame({'label':train_df.Sentiment,

                        'text':train_df.Phrase})
test_df = pd.DataFrame({'label':test_df.PhraseId,

    'text':test_df.Phrase})
lm_df = [train_df,test_df]

lm_df = pd.concat(lm_df)

lm_df.to_csv('./stuff/lm.csv',index=False)
from sklearn.model_selection import train_test_split



# split data into training and validation set

df_trn, df_val = train_test_split(train_df, stratify = train_df['label'], test_size = 0.1, random_state = SEED)
df_trn.shape, df_val.shape
# Language model data

data_lm = TextLMDataBunch.from_csv("",'./stuff/lm.csv',bs=128, min_freq=1)



# Classifier model data

data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, min_freq=1, vocab=data_lm.train_ds.vocab, bs=128)
# data_lm.show_batch()
learn = language_model_learner(data_lm,arch=AWD_LSTM, drop_mult=0.3)
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
lr = 7e-2

learn.fit_one_cycle(1, lr, moms=(0.8,0.7))
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, slice(2e-4,2e-3), moms=(0.8,0.7))
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, slice(5e-5,5e-4), moms=(0.8,0.7))
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, slice(5e-6,5e-5), moms=(0.8,0.7))
# learn.recorder.plot_lr(show_moms=True)
# learn.recorder.plot_metrics()
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas,arch=AWD_LSTM,drop_mult=0.3)
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(df_trn['label']),

                                                 df_trn['label'])
learn.loss_func.func = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().cuda())
kappa = KappaScore()
learn.metrics = [kappa, accuracy]
learn.load_encoder('ft_enc')

learn.freeze()
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
lr = 1e-1

learn.fit_one_cycle(1,lr, moms=(0.8,0.7))
learn.freeze_to(-2)
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(20,max_lr=slice(1e-2/2.6**4,1e-2), moms=(0.8,0.7))
learn.freeze_to(-3)
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(19,max_lr=slice(1e-4/2.6**4,1e-4), moms=(0.8,0.7))
# learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
# learn.fit_one_cycle(3,max_lr=slice(1e-5/2.6**4,1e-5), moms=(0.8,0.7))
# learn.save('final-model')
# interp = ClassificationInterpretation.from_learner(learn)

# losses,idxs = interp.top_losses()
# interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
# interp.most_confused(min_val=2)
learn.export()
test_loader = TextList.from_df(df=test_df,cols='text')
learn = load_learner(path='',file='export.pkl',test=test_loader,bs=128)
probs,_ = learn.get_preds(ds_type=DatasetType.Test,ordered=True)
submission_df = pd.read_csv('../input/sampleSubmission.csv')
submission_df.head()
submission_df['Sentiment'] = probs.argmax(1)
submission_df.to_csv('results.csv',index=False)