# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

from tqdm import  tqdm

tqdm.pandas()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/awd-lstm/"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.text import *

from sklearn.model_selection import train_test_split



MODEL_PATH = ''

INPUT_JIGSAW = '../input/jigsaw-unintended-bias-in-toxicity-classification/'

TEXT_COL = 'comment_text'

train = pd.read_csv(INPUT_JIGSAW + 'train.csv', index_col='id')

train['target'] = (train['target'] > 0.5).astype(int)

test = pd.read_csv(INPUT_JIGSAW + 'test.csv', index_col='id')



train_df, val_df = train_test_split(train, test_size=0.1)





del train

gc.collect()
# todo optimize runtime of tokenization



def shorten(text):

    x = text.split()

    x = x[:210]

    x = ' '.join(x)

    return x



train_df[TEXT_COL] = train_df[TEXT_COL].progress_apply(lambda x: shorten(x))

val_df[TEXT_COL] = val_df[TEXT_COL].progress_apply(lambda x: shorten(x))

test[TEXT_COL] = test[TEXT_COL].progress_apply(lambda x: shorten(x))



data_lm = TextClasDataBunch.from_df(MODEL_PATH,train_df,valid_df=val_df, test_df=test, text_cols=[TEXT_COL], label_cols=['target'])

#data_lm.save()
del train_df, val_df

gc.collect()
awd_lstm_clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.4,

                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
learner = text_classifier_learner(data_lm, AWD_LSTM, max_len=210,config=awd_lstm_clas_config, pretrained = False)
fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learner.load_pretrained(*fnames, strict=False)

learner.freeze()
# learner.lr_find(start_lr=1e-8, end_lr=1e2)

# learner.recorder.plot()
#learner.fit_one_cycle(1, 1e-3)
learner.unfreeze()

learner.fit_one_cycle(1, 1e-3)



# learner.fit_one_cycle(1, 1e-3)
oof = learner.get_preds(ds_type=DatasetType.Valid)
preds = learner.get_preds(ds_type=DatasetType.Test, ordered=True)
o = oof[0].cpu().data.numpy()

l = oof[1].cpu().data.numpy()
from sklearn.metrics import roc_auc_score, accuracy_score
accuracy_score(l,o[:,1]>0.5), roc_auc_score(l,o[:,1])
p = preds[0].cpu().data.numpy()
submission = pd.read_csv(INPUT_JIGSAW + 'sample_submission.csv', index_col='id')
submission['prediction'] = p[:,1]
submission.to_csv('submission.csv')
submission.head()