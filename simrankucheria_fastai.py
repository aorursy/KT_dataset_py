# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.text import *

from fastai import *
#path = Path('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/')

#path.ls()

!mkdir data

!pwd

!cp -a ../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv ./data/

!cp -a ../input/jigsaw-multilingual-toxic-comment-classification/test.csv ./data/

!cp -a ../input/jigsaw-multilingual-toxic-comment-classification/validation.csv ./data/

#!cp -a ../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv ./data

#!cp -a ../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv ./data

!ls data



path = Path('/kaggle/working/data/')

path.ls()



df_train = pd.read_csv(path/'jigsaw-toxic-comment-train.csv')

#df_test = pd.read_csv(path/'jigsaw-unintended-bias-train.csv', usecols=[0,1,2,3,4,5,6,7])

#df_test.columns=['id','comment_text','toxic','severe_toxic','obscene','identity_hate','insult','threat']

#df = df_train.append(df_test)

df = df_train

df.head()
bs = 64

data_lm = (TextList.from_df(df, path, cols='comment_text')

                .split_by_rand_pct(0.1)

                .label_for_lm()

                .databunch(bs=bs))
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))
learn.save_encoder('fine_tuned_enc')
test = pd.read_csv(path/"test.csv")

test_datalist = TextList.from_df(test, cols='content')
data_cls = (TextList.from_csv(path, 'jigsaw-toxic-comment-train.csv', cols='comment_text', vocab=data_lm.vocab)

                .split_by_rand_pct(valid_pct=0.1)

                .label_from_df(cols=['toxic'])

                .add_test(test_datalist)

                .databunch())

data_cls.save('data_clas.pkl')
data_clas = load_data(path, 'data_clas.pkl', bs=bs)

data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.save('first')

learn.load('first');
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.save('second');

learn.load('second');
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.save('third')

learn.load('third');
learn.unfreeze()

learn.fit_one_cycle(1, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
preds, target = learn.get_preds(DatasetType.Test, ordered=True)

labels = preds.numpy()

labels
labels
test_id = test['id']

label_cols = ['something','toxic']



submission = pd.DataFrame({'id': test_id})

#submission=pd.DataFrame(preds.numpy(), columns = label_cols)

submission = pd.concat([submission, pd.DataFrame(preds.numpy(), columns = label_cols)], axis=1)

header=["id","toxic"]

submission.to_csv('submission.csv',columns=header, index=False)

submission.head()