#!pip3 install fastai==1.0.42
import fastai
fastai.__version__
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
from fastai import *

from fastai.text import * 

from fastai.gen_doc.nbdoc import *

from fastai.datasets import * 

from fastai.datasets import Config

from pathlib import Path

import pandas as pd
path = Path("../input")
!ls '../input/'
!unzip '../input/*.zip' 
train_df = pd.read_csv('/kaggle/working/train.csv')#.sample(frac=0.01)

#train_df['is_valid'] = 0

train_df.head(5)
#train_df.loc[:50, 'is_valid'] = 1

#train_df.head()
test_df = pd.read_csv('/kaggle/working/test.csv')#.sample(frac=0.01)

#test_df['is_valid'] = 0

test_df.head(5)
from sklearn.model_selection import train_test_split



train_df, valid_df = train_test_split(train_df, test_size=0.2)
#data_lm = TextLMDataBunch.from_df(path, 

#                                   train_df,

#                                   valid_df,

#                                   text_cols='comment_text')
#??TextList.from_csv()
data_lm = (TextList.from_csv(path, '/kaggle/working/train.csv', cols='comment_text') 

                   .random_split_by_pct(0.3)

                   .label_for_lm()

                   .add_test(TextList.from_csv(path, '/kaggle/working/test.csv', cols='comment_text'))

                   .databunch())
data_lm
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, model_dir="/tmp/model/", drop_mult=0.3)

learn.model
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))

#learn.fit_one_cycle(1, 1e-1)

learn.save('mini_train_lm')

learn.save_encoder('mini_train_encoder')
#learn.unfreeze()

#learn.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))
learn.predict("Why the edits made under ", n_words=10)
learn.show_results()
data_clas = (TextList.from_csv(path, '/kaggle/working/train.csv',cols='comment_text', vocab=data_lm.vocab) #test='test'

    #.split_from_df(col='is_valid') #is_valid

    .random_split_by_pct(valid_pct=0.2)

    .label_from_df(cols=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])

    .add_test(TextList.from_csv(path, '/kaggle/working/test.csv', cols='comment_text'))

    .databunch(bs=42))
# Classifier model data

#data_clas = TextClasDataBunch.from_csv(path, 'train.csv', vocab=data_lm.train_ds.vocab, bs=32)
#from sklearn.model_selection import train_test_split



#train_df, valid_df = train_test_split(train_df, test_size=0.2)
#??TextClasDataBunch.from_df()
#data_clas = (TextClasDataBunch.from_df(path,train_df, valid_df, test_df, 

#                                   vocab=data_lm.vocab, 

#                                   text_cols='comment_text', 

#                                   label_cols=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])

             #.random_split_by_pct(valid_pct=0.2)

#             .split_from_df(col='is_valid') #is_valid

#             .databunch(bs=42) )
#data_clas.show_batch()
#data_clas
data_clas.c
learn = text_classifier_learner(data_clas, model_dir="/tmp/model/", drop_mult=0.5) #metrics=[accuracy_thresh]

learn.load_encoder('mini_train_encoder')

learn.freeze()

learn.model
learn.lr_find()
learn.recorder.plot()
#learn.crit = F.cross_entropy
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)
learn.metrics = metrics=[acc_02, f_score] #[top_k_accuracy]
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

#learn.fit_one_cycle(1, slice(1e-3,1e-2))

learn.save('mini_train_clas')
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.freeze_to(-2)

#learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))



learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
#learn.freeze_to(-3)

#learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
#learn.unfreeze()

#learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))



#learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
# get predictions

preds, targets = learn.get_preds()



predictions = np.argmax(preds, axis = 1)

pd.crosstab(predictions, targets)
learn.show_results()
probs, _ = learn.get_preds(DatasetType.Test,ordered=True)
#preds = probs.argmax(dim=1)
#preds.shape
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission = pd.DataFrame({'id': test_df["id"]})
submission = pd.concat([submission, pd.DataFrame(probs.numpy(), columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)

submission.head(n=10)