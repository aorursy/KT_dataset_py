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
PATH="../input/"
df_train = pd.read_csv(f'{PATH}train.csv')

df_test = pd.read_csv(f'{PATH}test.csv')

df_train.head()
import fastai.utils.collect_env; fastai.utils.collect_env.show_install(1)
import torch

print(torch.cuda.is_available())
%load_ext autoreload

%autoreload 2



%matplotlib inline
from fastai.imports import *

from fastai.tabular import * 



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics
df_train.head()
procs = [FillMissing, Normalize]

dep_var = 'target'



valid_idx = range(len(df_train)-2000, len(df_train))
valid_idx
df = df_train.drop('ID_code', axis=1)

#df_test = df_test.drop('ID_code', axis=1)
data = TabularDataBunch.from_df('.', df, dep_var, valid_idx=valid_idx, procs=procs, test_df=df_test.drop('ID_code', axis=1))
data.train_ds.cont_names
#(cat_x,cont_x),y = next(iter(data.train_dl))

#for o in (cat_x, cont_x, y): print(to_np(o[:5]))
learn1 = tabular_learner(data, layers=[200,100], ps=[0.5, 0.2], metrics=[accuracy, metrics.roc_auc_score])

learn1.fit_one_cycle(3, 1e-2)
learn2 = tabular_learner(data, layers=[200,100], ps=[0.5, 0.2], metrics=[accuracy, metrics.roc_curve])

learn2.fit(10, 1e-2)
learn = tabular_learner(data, layers=[200,100], ps=[0.5, 0.2], metrics=accuracy)

learn.fit(10, 1e-3)
#df_test.head()
#??learn.predict
#df.iloc[0]
#learn.predict(df_test)
#df_test.describe()
test_preds, _ = learn.get_preds(ds_type=DatasetType.Test)
result = to_np(test_preds[:,1])
result
learn.show_results()
df_test['target'] = result
df_test.head()
output = df_test[['ID_code', 'target']]
output.to_csv('submission.csv', index=False)