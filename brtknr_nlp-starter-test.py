!pip install -q torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
!pip install -q git+http://github.com/fastai/fastai
from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
print(torch.__version__)
print(torch.cuda.is_available())
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
df = pd.read_csv('../input/socialmedia_relevant_cols.csv', encoding="ISO-8859-1")
df[[0,1]] = df[['class_label', 'text']]
del df['text']
del df['choose_one']
del df['class_label']
df
df.head()
df.tail()
perm = np.random.permutation(len(df))
train_df = df.iloc[perm[:10000]]
valid_df = df.iloc[perm[10000:]]
len(train_df), len(valid_df)
data_lm = TextLMDataBunch.from_df(Path('.'), train_df, valid_df)
data_clas = TextClasDataBunch.from_df(Path('.'), train_df, valid_df, vocab=data_lm.train_ds.vocab)
learn = RNNLearner.language_model(data_lm, pretrained_model=URLs.WT103)
learn.unfreeze()
learn.fit(2, slice(1e-4,1e-2))
learn.save_encoder('enc')
learn = RNNLearner.classifier(data_clas)
learn.load_encoder('enc')
learn.fit(3, 1e-3)

