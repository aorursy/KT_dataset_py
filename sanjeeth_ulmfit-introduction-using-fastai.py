import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
train = pd.read_excel('../input/Train_Data.xlsx')
import fastai

from fastai.text import *

from fastai.callbacks import *
# check the contents of the dat set

train.head()
train['question_text'][2]
train['question_text'][21]
from sklearn.model_selection import train_test_split

train, val = train_test_split(train)
# Language model data bunch

data_lm = TextLMDataBunch.from_df('.', train,val,text_cols='question_text',label_cols='target')
#save the preprocessed data

data_lm.save()
# Classifier model data

data_clas  = TextClasDataBunch.from_df('.', train_df=train,text_cols='question_text',label_cols='target',valid_df=val,vocab=data_lm.train_ds.vocab)
data_clas.save()
data_clas.show_batch()
data_clas.vocab.itos[:10]
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3,pretrained=True)

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, 5.75E-02,callbacks=[SaveModelCallback(learn, name="best_lm")], moms=(0.8,0.7))
learn.save('fit_head')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(3,3.98E-04,callbacks=[SaveModelCallback(learn, name="best_lm")], moms=(0.8,0.7))
learn.load('best_lm')
learn.save_encoder('AIBoot_enc')
learn1 = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
learn1.load_encoder('AIBoot_enc')
learn1.lr_find()
learn1.recorder.plot(suggestion=True)
best_clf_lr = learn1.recorder.min_grad_lr

best_clf_lr
learn1.fit_one_cycle(1, best_clf_lr)
learn1.freeze_to(-2)
learn1.fit_one_cycle(1, best_clf_lr)
learn1.unfreeze()
learn1.lr_find()

learn1.recorder.plot(suggestion=True)
learn1.fit_one_cycle(3, 2e-3)
learn1.show_results()
learn.predict('Is it just me or have you ever been')
'Is it just me or have you ever been in this phase wherein you became ignorant to the people you once loved, completely disregarding their feelings/lives so you get to have something go your way and feel temporarily at ease. How did things change?'