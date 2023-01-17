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
from fastai.text import *
data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',index_col='id')

data.head(5)
data.shape
cols=['text']

data_bunch = (TextList.from_df(data, cols=cols)

                .split_by_rand_pct(0.2)

                .label_for_lm()  

                .databunch(bs=48))

data_bunch.show_batch()
learn = language_model_learner(data_bunch,AWD_LSTM,pretrained_fnames=['/kaggle/input/wt103-fastai-nlp/lstm_fwd','/kaggle/input/wt103-fastai-nlp/itos_wt103'],pretrained=True,drop_mult=0.7)



# select the appropriate learning rate

learn.lr_find()



# we typically find the point where the slope is steepest

learn.recorder.plot()



# Fit the model based on selected learning rate

learn.fit_one_cycle(2, 1e-3, moms=(0.8,0.7))

learn.unfreeze()

learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))



# Save the encoder for use in classification

learn.save_encoder('fine_tuned_enc')
train=data[:6000]

val=data[6000:]

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',index_col='id')
target_cols=['target']

data_clas = TextClasDataBunch.from_df('.', train, val, test,

                  vocab=data_bunch.vocab,

                  text_cols=cols,

                  label_cols=target_cols,

                  bs=32)
learn_classifier = text_classifier_learner(data_clas, AWD_LSTM,pretrained=False,drop_mult=0.7,metrics=[accuracy])

fnames = ['/kaggle/input/wt103-fastai-nlp/lstm_fwd.pth','/kaggle/input/wt103-fastai-nlp/itos_wt103.pkl']

learn_classifier.load_pretrained(*fnames, strict=False)

# load the encoder saved  

learn_classifier.load_encoder('fine_tuned_enc')

learn_classifier.freeze()



# select the appropriate learning rate

learn_classifier.lr_find()



# we typically find the point where the slope is steepest

learn_classifier.recorder.plot()

import warnings

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

learn_classifier.fit_one_cycle(15, 1e-3, moms=(0.8,0.7))
preds_test, target_test = learn_classifier.get_preds(DatasetType.Test, ordered=True)

y = torch.argmax(preds_test, dim=1)

y.numpy().shape
submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print(submission.shape)

submission['target']=y.numpy()

submission.head()
submission['target'].value_counts()
submission.to_csv('submission.csv',index=False)