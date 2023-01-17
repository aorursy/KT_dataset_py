import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from fastai import *
from fastai.text import *
import os

path = '/kaggle/input/hindi-poetry-dataset'; path = Path(path)
file = 'sample_naaraz_22.txt'; file = Path(file)
filename = path/file
f = open(filename, 'r')
text = f.read()
print(text[:1000])
poems = []
for file in os.listdir(path):
    filename = path/file
    f = open(filename, 'r')
    text = f.read()
    poems.append(text)

poems_df = pd.DataFrame(poems)
poems_df.head()
data_lm = (TextList.from_df(poems_df)
                  .split_by_rand_pct(0.2)
                  .label_for_lm()
                  .databunch(bs=128))
data_lm.show_batch(rows=6)
learn = language_model_learner(data_lm, AWD_LSTM, metrics=[accuracy,Perplexity()], drop_mult=0.5)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, 1e-01, moms=(0.8,0.7))
predict = learn.predict('तुम्हें', n_words = 50)
print(predict)
predict = learn.predict('मैं', n_words = 150)
print(predict)