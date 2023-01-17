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
pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")[["text","target"]].to_csv("train2.csv",index=False)

pd.read_csv("/kaggle/working/train2.csv")
data_lm = TextLMDataBunch.from_csv("/kaggle/working/","train2.csv",bs=128,text_cols=0, label_cols=1)

data_clas = TextClasDataBunch.from_csv("/kaggle/working/","train2.csv", vocab=data_lm.train_ds.vocab, bs=128,text_cols=0, label_cols=1)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

learn.fit_one_cycle(1, 1e-2)
learn.fit_one_cycle(10, 1e-3)
learn.unfreeze()

learn.fit_one_cycle(3, 1e-3)
learn.save_encoder('ft_enc')

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc')
learn.fit_one_cycle(3, 1e-2)

learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
learn.unfreeze()

learn.fit_one_cycle(10, slice(2e-3/100, 2e-3))
test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test['text']
learn.data.add_test(test['text'].to_list())
preds,y = learn.get_preds(ds_type=DatasetType.Test)

labels = np.argmax(preds, 1)
test['target'] = labels.numpy()
test[['id','target']].to_csv('submission.csv',index=False)