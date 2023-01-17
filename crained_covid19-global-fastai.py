# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_extd_ext autoreload

%autoreload 2

%matplotlib inline
# This file contains all the main external libs we'll use

from fastai import *

from fastai.tabular import *

from sklearn.metrics import recall_score
path = Path('/kaggle/input/')
path.ls()
path = Path('/kaggle/input/covid19-global-forecasting-week-1')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv(path/'train.csv')
df_test = pd.read_csv(path/'test.csv')
len(df.index)
df.head(5)
add_datepart(df, 'Date')
add_datepart(df_test, 'Date')
dep_var = ['ConfirmedCases', 'Fatalities']

#cat_names = ['Province/State', 'Country/Region']

cat_names = ['Country/Region']

cont_names = ['Lat', 'Long', 'Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear' ]

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(df.iloc[14000:-1].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(list(range(1000,13000)))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, metrics=[rmse, accuracy])
learn.model
learn.model_dir = "/kaggle/working"
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 1e-6, wd=0.2)

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-6)