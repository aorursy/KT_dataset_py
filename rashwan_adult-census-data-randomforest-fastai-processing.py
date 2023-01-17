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

from sklearn.ensemble import RandomForestClassifier
PATH = Path('../input')

PATH.ls()
df_raw = pd.read_csv(PATH/"adult.csv")
dep_var = 'income'

cat_var = ['workclass','education','marital.status','occupation','relationship','race','sex','native.country']

cont_var = ['education.num', 'hours.per.week', 'age', 'capital.loss', 'fnlwgt', 'capital.gain']

proc = [FillMissing,Categorify,Normalize]

valid_idx = range((len(df_raw) - 6000),len(df_raw))
data = (TabularList.from_df(df_raw,cat_names=cat_var,cont_names=cont_var,procs=proc)

                    .split_by_idx(valid_idx)

                    .label_from_df(cols=dep_var)

                    .databunch())
(data.train_ds.x.conts.shape,data.train_ds.x.codes.shape)
x_train = np.concatenate((data.train_ds.x.conts,data.train_ds.x.codes),axis=1)
y_train = to_data(list(data.train_ds.y))
x_valid = np.concatenate((data.valid_ds.x.conts,data.valid_ds.x.codes),axis=1)

x_valid.shape
y_valid = to_data(list(data.valid_ds.y))
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(x_train), y_train), rmse(m.predict(x_valid), y_valid),

                m.score(x_train, y_train), m.score(x_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestClassifier(n_jobs=-1,n_estimators=100,min_samples_leaf=5,max_features='log2',)

m.fit(x_train,y_train)

print_score(m)
print("Validation accuracy:{:.3f}".format(m.score(x_valid,y_valid)))