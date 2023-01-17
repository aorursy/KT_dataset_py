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
from fastai.tabular import *

import path
PATH = Path('../input')

PATH.ls()
df_raw = pd.read_csv(PATH/"adult.csv")
print(df_raw.shape)

df_raw.dtypes
dep_var = 'income'

cat_var = ['workclass','education','marital.status','occupation','relationship','race','sex','native.country']

cont_var = ['education.num', 'hours.per.week', 'age', 'capital.loss', 'fnlwgt', 'capital.gain']

proc = [FillMissing,Categorify,Normalize]

valid_idx = range((len(df_raw) - 6000),len(df_raw))
data = (TabularList.from_df(df_raw,cat_names=cat_var,cont_names=cont_var,procs=proc)

                    .split_by_idx(valid_idx)

                    .label_from_df(cols=dep_var)

                    .databunch())
data.show_batch()
learner = tabular_learner(data,layers = [200,100],metrics=accuracy)
learner
learner.fit_one_cycle(5,1e-2)
learner.lr_find()
learner.recorder.plot()
learner.unfreeze()
learner.fit_one_cycle(5)