from fastai import *

from fastai.text import *

from pathlib import Path

import pandas as pd

import numpy as np

import re



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score





import torch

print("Cuda available" if torch.cuda.is_available() is True else "CPU")

print("PyTorch version: ", torch.__version__)
train_df = pd.read_csv('../input/train.csv', nrows=100)

test_df = pd.read_csv('../input/test.csv', nrows=10)
def clean_text(text):

    return (text.str.lower()

                         .str.replace('\?+', ' ?')

                         .str.replace('\!+', ' !')

                         .str.replace('#', '# ')

                         .str.replace('@', '@ ')

                         .str.replace(':\)', '>')

                         .str.replace('won\'t', 'will not')

                         .str.replace('can\'t', 'can not')

                         .str.replace('it\'s', 'it is')

                         .str.replace('that\'s', 'that is')

                         .str.replace('\'s', '')

                         .str.replace('n\'t', ' not')

                         .str.replace('\'re', ' are')

                         .str.replace('\'d', ' would')

                         .str.replace('\'ll', ' will')

                         .str.replace('\'t', ' not')

                         .str.replace('\'ve', ' have')

                         .str.replace('\'m', ' am')

                         .str.replace(r'sh\*tty', 'shit')

                         .str.replace('[\'\":();,.\-—/_]', ' ')

                         .str.replace(r'(ha|hha|hhha)+', 'ha')

                         .str.replace(r'\bur\b', 'you are')

                         .str.replace(r'f+u+', 'fu')

                         .str.replace(r'\*', '')

                         .str.replace(r'%', ' %')

                         .str.replace(' iv ', ' 4 ')

                         .str.replace(' cc ', ' civil comments ')

                         .str.replace(' ww ', ' willamette week ')

                         .str.replace(r'\$+', '$ ')

                         .str.replace('&', ' and ')

                         .str.replace(' os x ', ' osx ')

                         .str.replace('\s+', ' ')

).str.strip()
train_df['comment_text_clean'] = clean_text(train_df['comment_text'])

print('train is done')



test_df['comment_text_clean'] = clean_text(test_df['comment_text'])

print('test is done')
len(train_df), len(test_df)
train_df['target_round'] = (train_df['identity_annotator_count'] >= 2) & (train_df['target'] >= 0.4)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['target_round'])
data_lm = TextLMDataBunch.from_df(

    path='',

    train_df=train_df,

    valid_df=val_df,

    test_df=test_df,

    text_cols=['comment_text_clean'],

    label_cols=['target_round'],

    #label_cols=['target_better'],

    #classes=['target_better'],

    min_freq=3

)
learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.8)
learn.lr_find(start_lr=1e-6, end_lr=1e2)

learn.recorder.plot()
learn.fit_one_cycle(cyc_len=3, max_lr=1e-01)
learn.unfreeze()

learn.fit_one_cycle(cyc_len=10, max_lr=1e-3, moms=(0.8, 0.7))
learn.save_encoder('ft_enc')
data_class = TextClasDataBunch.from_df(

    path='',

    train_df=train_df,

    valid_df=val_df,

    test_df=test_df,

    text_cols=['comment_text_clean'],

    label_cols=['target_round'],

    #label_cols=['target_better'],

    min_freq=3,

    vocab=data_lm.train_ds.vocab,

    #label_delim=' '

)
learn = text_classifier_learner(data_class, arch=AWD_LSTM, drop_mult=0.8)

learn.load_encoder('ft_enc')

learn.freeze()
learn.lr_find(start_lr=1e-8, end_lr=1e2)

learn.recorder.plot()
learn.fit_one_cycle(cyc_len=3, max_lr=1e-005)
oof = learn.get_preds(ds_type=DatasetType.Valid)

o = oof[0]

l = oof[1]



accuracy_score(l,o[:,1]>0.5), roc_auc_score(l,o[:,1])
learn.freeze_to(-2)

learn.fit_one_cycle(3, slice(1e-4,1e-2))
oof = learn.get_preds(ds_type=DatasetType.Valid)

o = oof[0]

l = oof[1]



accuracy_score(l,o[:,1]>0.5), roc_auc_score(l,o[:,1])
learn.freeze_to(-3)

learn.fit_one_cycle(3, slice(1e-5,5e-3))
oof = learn.get_preds(ds_type=DatasetType.Valid)

o = oof[0]

l = oof[1]



accuracy_score(l,o[:,1]>0.5), roc_auc_score(l,o[:,1])
learn.unfreeze()

learn.fit_one_cycle(10, slice(1e-5,1e-3))
oof = learn.get_preds(ds_type=DatasetType.Valid)

o = oof[0]

l = oof[1]



accuracy_score(l,o[:,1]>0.5), roc_auc_score(l,o[:,1])
preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
p = preds[0][:,1]
test_df['prediction'] = p
test_df.sort_values('prediction', inplace=True)

test_df.reset_index(drop=True, inplace=True)
ii = 9993

print(test_df['comment_text_clean'][ii])

print(test_df['prediction'][ii])
train_df['comment_text'][4595]
train_df[train_df['target'] > 0.005].sort_values('target')[['comment_text', 'target']]