# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%reload_ext autoreload

%autoreload 2

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.text import *





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings('ignore')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/quotes-500k/quotes.csv");df
df = df[["quote"]].dropna()
df.shape
data = (TextList.from_df(df)

                .split_by_rand_pct(0.1)

                .label_for_lm()           

                .databunch(bs=128))
data.show_batch()
len(data.train_dl),len(data.valid_dl)

learn = language_model_learner(data, AWD_LSTM, drop_mult=0.3).to_fp16()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, 5e-2)
TEXT = "think twice"

N_WORDS = 30

N_SENTENCES = 10

print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.3) for _ in range(N_SENTENCES)))
learn.save("stage-1")
learn.load("stage-1");
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))

learn.save("stage-2")
learn.fit_one_cycle(3, 1e-3, moms=(0.8,0.7))
texts = ["think twice","do not","why is it","life is a","the sea", "oh","you can","dream big"]

for e in texts:

    TEXT = e

    N_WORDS = 30

    N_SENTENCES = 3

    print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.5) for _ in range(N_SENTENCES)))