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

from fastai import *
path = Path('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/')
!mkdir data

!cp -a {path}/*.* ./data/

!ls data
!rm data/jigsaw-toxic-comment-train-processed-seqlen128.csv
!rm data/jigsaw-unintended-bias-train-processed-seqlen128.csv
path = Path('/kaggle/working/data/')

path.ls()
df=pd.read_csv(path/'jigsaw-toxic-comment-train.csv')

df.head()
bs=64

data_lm = (TextList.from_df(df, path, cols='comment_text')

                .split_by_rand_pct(0.2)

                .label_for_lm()

                .databunch(bs=bs))
learn=language_model_learner(data_lm,AWD_LSTM,drop_mult=0.3,metrics=[accuracy])
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(2,1e-02,moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(4,1e-03,moms=(0.8,0.7))
learn.save_encoder('fine_tuned_enc')
TEXT="My name is"

N_WORDS=50

N_SENTENCES=2

print("\n".join(learn.predict(TEXT,N_WORDS,temperature=0.75) for _ in range(N_SENTENCES)))
test = pd.read_csv(path/"test.csv")

test_datalist = TextList.from_df(test, cols='content')

data_cls = (TextList.from_csv(path, 'jigsaw-toxic-comment-train.csv', cols='comment_text', vocab=data_lm.vocab)

                .split_by_rand_pct(valid_pct=0.2)

                .label_from_df(cols=['toxic', 'severe_toxic','obscene', 'threat', 'insult', 'identity_hate'], label_cls=MultiCategoryList, one_hot=True)

                .add_test(test_datalist)

                .databunch())
learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.5,metrics=[accuracy])
learn.load_encoder('fine_tuned_enc')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1,4e-02,moms=(0.8,0.7))
learn.save('stage-1')
learn.load('stage-1')
learn.freeze_to(-2)

learn.fit_one_cycle(1,slice(1e-02/(2.6**4),1e-02),moms=(0.8,0.7))
learn.save('stage-2')
learn.load('stage-2')
learn.unfreeze()

learn.fit_one_cycle(2,slice(1e-03/(2.6**4),1e-03),moms=(0.8,0.7))
learn.save('stage-3')
preds, target = learn.get_preds(DatasetType.Test, ordered=True)

labels = preds.numpy()
test_id = test['id']

label_cols = ['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']



sub = pd.DataFrame({'id': test_id})

sub = pd.concat([sub, pd.DataFrame(preds.numpy(), columns = label_cols)], axis=1)

sub.head()
submission=pd.read_csv(path/'sample_submission.csv')

submission['toxic']=sub['toxic']
submission.to_csv('submission.csv',index=False)