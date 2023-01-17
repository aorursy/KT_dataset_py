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
from fastai.tabular import *
from fastai.callbacks import *

path = "/kaggle/input/banksorted/bank-full-SORTED.csv"
df = pd.read_csv(path)
df = df.assign(outcome=(df['y'] == 'yes').astype(int))
#df.head(8000)
train = df.iloc[0:10000]
train = train.sample(frac=1).reset_index(drop=True)
train.head(20)


dep_var = 'outcome'
cat_names = ['job', 'marital', 'education', 'housing', 'loan', 'month', 'poutcome', 'contact']
cont_names = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
procs = [FillMissing, Categorify, Normalize]



test = TabularList.from_df(train.iloc[7000:9999].copy(), path=path, cat_names=cat_names, cont_names=cont_names)



data = (TabularList.from_df(train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                            .split_by_idx(list(range(100,7000)))
                            .label_from_df(cols=dep_var)
                            .add_test(test)
                            .databunch())


data.show_batch(rows=20)


learn = tabular_learner(data, layers=[16,4], ps=[0.001,0.01], emb_drop=0.04, metrics=accuracy)
learn.model_dir="/kaggle/working"
learn.lr_find()
learn.recorder.plot(moms=True)
learn.fit(2)
learn.recorder.plot_losses() 
learn.recorder.plot_metrics()
print(learn)
row = df.iloc[9]
df.iloc[7000]
learn.predict(df.iloc[7000])
#data